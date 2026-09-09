#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spatial_axis_gat.config import load_config, resolve_path
from spatial_axis_gat.data import read_adata
from spatial_axis_gat.labelme import parse_batch_annotations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    manifest = pd.read_csv(resolve_path(cfg, cfg["labelme"]["manifest_csv"]))
    json_dir = resolve_path(cfg, cfg["labelme"]["json_dir"])
    image_dir = resolve_path(cfg, cfg["labelme"]["image_dir"])
    adata = read_adata(cfg["data"]["h5ad_path"], backed="r")
    batch_key = cfg["data"]["batch_key"]
    spatial_key = cfg["data"]["spatial_key"]
    epi_key = cfg["data"]["epithelial_existing_key"]

    all_rows = []
    for _, row in manifest.iterrows():
        batch = str(row["batch"])
        json_name = str(row["image_name"]).replace(".png", ".json")
        json_path = json_dir / json_name
        if not json_path.exists():
            fallback = image_dir / json_name
            json_path = fallback if fallback.exists() else json_path
        if not json_path.exists():
            print(f"Skipping {batch}: no LabelMe JSON found at {json_path}")
            continue

        global_idx = np.flatnonzero(np.asarray(adata.obs[batch_key].astype(str) == batch))
        coords = np.asarray(adata.obsm[spatial_key][global_idx])
        epi = np.asarray(adata.obs[epi_key].iloc[global_idx], dtype=float)
        parsed = parse_batch_annotations(
            json_path=json_path,
            manifest_row=row,
            coords=coords,
            existing_epithelial=epi,
            batch=batch,
            villus_labels=set(cfg["labelme"]["villus_labels"]),
            bottom_labels=set(cfg["labelme"]["bottom_labels"]),
        )
        parsed["global_index"] = global_idx[parsed["local_cell_index"].to_numpy()]
        all_rows.append(parsed)
        print(f"Parsed {len(parsed)} labeled cells from {json_path}")

    if not all_rows:
        raise SystemExit("No annotations were parsed. Add LabelMe JSON files before running this step.")

    labels = pd.concat(all_rows, ignore_index=True)
    out = resolve_path(cfg, cfg["labelme"]["parsed_labels_csv"])
    out.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(out, index=False)
    print(f"Wrote {out}")
    print(labels.groupby("training_villus_id").size().describe())


if __name__ == "__main__":
    main()

