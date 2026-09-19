"""Portable section-aware training and exact two-hop minibatch inference.

Uses only native PyTorch/PyG operations: no torch-sparse or pyg-lib build.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
from scipy import sparse
import torch

from .graph import build_batch_spatial_edge_index
from .metrics import regression_metrics
from .model import GATModel

FORMAT = "tissuemapper-graph/v1"  # Stable checkpoint identifier for existing models.


def provenance(path, args, inputs, outputs):
    def record(filename):
        filename = Path(filename).resolve()
        digest = hashlib.sha256()
        with filename.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        return dict(path=str(filename), sha256=digest.hexdigest(), bytes=filename.stat().st_size)
    Path(path).write_text(json.dumps(dict(
        timestamp_utc=datetime.now(timezone.utc).isoformat(), command=sys.argv,
        configuration=vars(args), python=platform.python_version(),
        packages={name: importlib.metadata.version(name) for name in ['spatialtrace-graph', 'torch', 'torch-geometric', 'numpy', 'scikit-learn']},
        inputs=[record(p) for p in inputs], outputs=[record(p) for p in outputs]), default=str, indent=2))


def classification_metrics(target, logits, threshold):
    from sklearn.metrics import average_precision_score, roc_auc_score
    probability = torch.as_tensor(logits).sigmoid().numpy()
    labels = np.asarray(target) >= .5
    both = len(np.unique(labels)) == 2
    return dict(n=len(labels), auroc=float(roc_auc_score(labels, probability)) if both else None,
                average_precision=float(average_precision_score(labels, probability)) if labels.any() else None,
                accuracy=float(np.mean((probability >= threshold) == labels)), threshold=threshold)


def read_input(path, feature_key="X_scVI", spatial_key="X_spatial", section_key="section_id"):
    adata = anndata.read_h5ad(path)
    if not adata.obs_names.is_unique:
        raise ValueError("Cell IDs (AnnData obs_names) must be unique")
    x = np.asarray(adata.obsm[feature_key], dtype=np.float32)
    coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)
    if adata.obs[section_key].isna().any():
        raise ValueError("Section IDs cannot be missing")
    sections = adata.obs[section_key].astype(str).to_numpy()
    if x.ndim != 2 or len(x) != len(adata) or not np.isfinite(x).all():
        raise ValueError("Features must be a finite dense (cells, features) matrix")
    if len(x) == 0:
        raise ValueError("Input contains no cells")
    return adata, torch.from_numpy(x), coords, sections


class Neighborhoods:
    """Exact incoming two-hop neighborhoods, including isolated-cell self loops."""
    def __init__(self, edge_index, num_nodes):
        edge = edge_index.cpu().numpy()
        self.matrix = sparse.csr_matrix((np.ones(edge.shape[1], dtype=np.int8),
            (edge[1], edge[0])), shape=(num_nodes, num_nodes))

    def batch(self, roots):
        roots = np.asarray(roots, dtype=np.int64)
        nodes = np.unique(roots)
        for _ in range(2):
            nodes = np.union1d(nodes, self.matrix[nodes].indices)
        local = self.matrix[nodes][:, nodes].tocoo()
        edges = torch.as_tensor(np.stack([local.col, local.row]), dtype=torch.long)
        return nodes, edges, torch.as_tensor(np.searchsorted(nodes, roots))


def load_checkpoint(path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != FORMAT:
        raise ValueError("Use a SpatialTRACE-Graph release checkpoint or train output")
    cfg = payload["model_config"]
    if payload.get("task", "coordinate") == "peyer":
        from .peyer import BinaryGAT
        model = BinaryGAT(**cfg)
    else:
        model = GATModel(**cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.to(device).eval(), payload


def predict_nodes(model, x, neighborhoods, *, indices=None, batch_size=256, device="cpu"):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    indices = np.arange(len(x)) if indices is None else np.asarray(indices)
    values = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            nodes, edges, roots = neighborhoods.batch(indices[start:start + batch_size])
            pred = model(x[nodes].to(device), edges.to(device))[roots.to(device)]
            values.append(pred.cpu().numpy())
    return np.concatenate(values) if values else np.empty(0, dtype=np.float32)


def predict(args):
    output = Path(args.output)
    report = output.with_suffix('.provenance.json')
    if output.exists() or report.exists():
        raise FileExistsError(output if output.exists() else report)
    adata, x, coords, sections = read_input(args.input, args.feature_key, args.spatial_key, args.section_key)
    model, payload = load_checkpoint(args.checkpoint, args.device)
    expected = payload["feature_space"]
    actual = adata.uns.get("tissuemapper_feature_space")
    if actual != expected:
        raise ValueError(f"Incompatible feature space: checkpoint expects {expected!r}, input declares {actual!r}. "
                         "Use its original fitted expression encoder, or retrain on your own features. "
                         "Do not relabel an independently learned embedding to bypass this check.")
    if x.shape[1] != payload["model_config"]["in_features"]:
        raise ValueError("Feature dimension differs from checkpoint")
    graph = payload["graph"]
    edges = build_batch_spatial_edge_index(coords, sections, **graph)
    if "feature_mean" in payload:
        x = (x - torch.as_tensor(payload["feature_mean"])) / torch.as_tensor(payload["feature_std"])
    values = predict_nodes(model, x, Neighborhoods(edges, len(x)), batch_size=args.batch_size, device=args.device)
    column = payload["output_column"]
    if payload.get("task") == "peyer":
        values = torch.from_numpy(values).sigmoid().numpy()
    output.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame({"cell_id": adata.obs_names, "section_id": sections,
                          "x": coords[:, 0], "y": coords[:, 1], column: values})
    if payload.get("task") == "peyer":
        result["peyer_call"] = values >= payload["threshold"]
    result.to_csv(output, index=False)
    provenance(report, args, [args.input, args.checkpoint], [output])
    print(f"Wrote {len(result)} predictions to {output}")


def train(args):
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        raise ValueError("Epochs, batch size and patience must be positive")
    if not 0 < args.threshold < 1:
        raise ValueError("Classifier threshold must lie strictly between 0 and 1")
    destination = Path(args.output_dir)
    if destination.exists():
        raise FileExistsError(destination)
    adata, x, coords, sections = read_input(args.input, args.feature_key, args.spatial_key, args.section_key)
    feature_space = adata.uns.get("tissuemapper_feature_space")
    if not isinstance(feature_space, str) or not feature_space:
        raise ValueError("Set uns['tissuemapper_feature_space'] to an identifier for your fitted feature encoder")
    split = adata.obs[args.split_key].astype(str).to_numpy()
    if not set(split) <= {"train", "validation", "test"}:
        raise ValueError("Split must contain only train, validation, or test")
    for section in np.unique(sections):
        if len(np.unique(split[sections == section])) != 1:
            raise ValueError(f"Section {section} crosses supervised splits")
    y = torch.tensor(adata.obs[args.target].to_numpy(dtype=np.float32))
    valid = np.isfinite(y.numpy())
    if ((y.numpy()[valid] < 0) | (y.numpy()[valid] > 1)).any():
        raise ValueError("Finite coordinate or probability labels must be between 0 and 1")
    groups = {s: np.flatnonzero((split == s) & valid) for s in ["train", "validation", "test"]}
    if len(groups["train"]) < 1 or len(groups["validation"]) < 1:
        raise ValueError("Training and validation sections each need finite labels")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    graph = {"n_neighbors": args.neighbors, "neighbor_policy": "knn"}
    edges = build_batch_spatial_edge_index(coords, sections, **graph)
    neighborhoods = Neighborhoods(edges, len(x))
    cfg = dict(in_features=x.shape[1], hidden_features=args.hidden_features,
               n_heads=args.heads, dropout=args.dropout)
    preprocessing = {}
    if args.task == 'peyer':
        from .peyer import BinaryGAT
        # Fit normalization on training sections only, including unlabeled neighbors.
        training_features = x[split == 'train']
        mean = training_features.mean(0)
        std = training_features.std(0, unbiased=False).clamp_min(1e-6)
        x = (x - mean) / std
        preprocessing = dict(feature_mean=mean, feature_std=std, threshold=args.threshold)
        model = BinaryGAT(**cfg).to(args.device)
        loss_function = torch.nn.functional.binary_cross_entropy_with_logits
        loss_name = 'bce'
    else:
        cfg['out_features'] = 1
        model = GATModel(**cfg).to(args.device)
        loss_function = torch.nn.functional.mse_loss
        loss_name = 'mse'
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best, best_epoch, best_state = float("inf"), 0, None
    history = []
    destination.mkdir(parents=True, exist_ok=False)
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(groups["train"])
        total = 0.0
        for start in range(0, len(order), args.batch_size):
            ids = order[start:start + args.batch_size]
            nodes, local_edges, roots = neighborhoods.batch(ids)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x[nodes].to(args.device), local_edges.to(args.device))[roots.to(args.device)]
            loss = loss_function(pred, y[ids].to(args.device))
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            total += loss.item() * len(ids)
        validation = predict_nodes(model, x, neighborhoods, indices=groups["validation"],
                                   batch_size=args.batch_size, device=args.device)
        val_loss = float(loss_function(torch.from_numpy(validation), y[groups['validation']]))
        history.append({"epoch": epoch, 'train_' + loss_name: total / len(order), 'validation_' + loss_name: val_loss})
        print(json.dumps(history[-1]), flush=True)
        if val_loss < best:
            best, best_epoch = val_loss, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch - best_epoch >= args.patience:
            break
    payload = {"format": FORMAT, "task": args.task, "model_config": cfg,
               "model_state_dict": best_state, "graph": graph, "feature_space": feature_space,
               "output_column": 'peyer_probability' if args.task == 'peyer' else "predicted_" + args.target,
               "selection": 'minimum validation ' + loss_name.upper(),
               "best_epoch": best_epoch, "seed": args.seed, **preprocessing}
    torch.save(payload, destination / "model.pt")
    model.load_state_dict(best_state, strict=True)
    metrics = {}
    for name, ids in groups.items():
        if len(ids):
            prediction = predict_nodes(model, x, neighborhoods, indices=ids,
                                        batch_size=args.batch_size, device=args.device)
            metrics[name] = (classification_metrics(y.numpy()[ids], prediction, args.threshold)
                             if args.task == 'peyer' else regression_metrics(y.numpy()[ids], prediction))
    pd.DataFrame(history).to_csv(destination / "history.csv", index=False)
    pd.DataFrame({"cell_id": adata.obs_names, "section_id": sections, "split": split,
                  "label_available": valid}).to_csv(destination / "split.csv", index=False)
    (destination / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    provenance(destination/'provenance.json', args, [args.input],
               [destination/name for name in ['model.pt', 'history.csv', 'split.csv', 'metrics.json']])
    print(f"Selected epoch {best_epoch}; wrote {destination / 'model.pt'}")


def create_demo(output):
    """Synthetic, labeled tutorial: three independent sections, no real data."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    rng = np.random.default_rng(17)
    coords = rng.uniform(0, 1, (96, 2)).astype(np.float32)
    features = np.column_stack([coords, coords ** 2, np.sin(coords * np.pi),
                               rng.normal(0, .1, (96, 2))]).astype(np.float32)
    obs = pd.DataFrame({"section_id": np.repeat(["section_a", "section_b", "section_c"], 32),
                        "split": np.repeat(["train", "validation", "test"], 32),
                        "crypt_villus": coords[:, 1], "epithelial_distance": coords[:, 0],
                        "peyer_label": (coords[:, 0] > .5).astype(float)},
                       index=[f"synthetic_{i:03}" for i in range(96)])
    data = anndata.AnnData(features, obs=obs)
    data.obsm["X_scVI"] = features  # Column name only: these are explicitly synthetic features.
    data.obsm["X_spatial"] = coords
    data.uns["tissuemapper_feature_space"] = "synthetic-tutorial-v1"
    data.uns["description"] = "Synthetic software test; not scVI and not paper performance"
    output.parent.mkdir(parents=True, exist_ok=True)
    data.write_h5ad(output)
    print(output)
