# Installation

Install uv using the [official installation instructions](https://docs.astral.sh/uv/getting-started/installation/). From a TissueMapper-Graph checkout:

```bash
uv sync --locked --extra cpu --extra dev
uv run --locked --extra cpu tissuemapper-graph --help
```

The .python-version file selects Python 3.12. uv.lock pins the environment; --locked refuses unintended dependency resolution changes. Keep the same torch extra on subsequent uv run commands.

For Linux/NVIDIA, choose --extra cu126 instead of cpu and --device cuda in model commands. A compatible NVIDIA driver is required; uv installs PyTorch's runtime libraries. Mac and Windows GPU acceleration are not validated. CPU installation does not require CUDA.

Optional extras: dev (tests/build), docs (MkDocs), figures (paper rendering), and expression (scVI workflows). Train a feature encoder on your own data and retain it for later inference. No pretrained graph model or paper-specific encoder is required or distributed. Paper rendering uses frozen results and does not refit UMAP.

See [uv's PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/) for index/extra behavior. Never synchronize an unrelated active environment; run from this checkout.
