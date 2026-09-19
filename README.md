# SpatialTRACE-Graph

TRACE stands for Tissue Region and Axis Coordinate Estimation.

Map tissue organization from spatial transcriptomics. Train graph attention models to learn anatomical coordinates or identify tissue regions from expression features, spatial neighbors, and your annotations.

## Install

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
gh repo clone maximilian-heeg/TissueMapper-Graph SpatialTRACE-Graph
cd SpatialTRACE-Graph
uv sync --locked --extra cpu
```

For an NVIDIA GPU on Linux, use `--extra cu126` instead of `--extra cpu`.

## Try it

Create a small synthetic dataset, train a coordinate model, and predict:

```bash
uv run --locked --extra cpu spatialtrace-graph create-demo \
  --output runs/demo/data.h5ad

uv run --locked --extra cpu spatialtrace-graph train \
  --input runs/demo/data.h5ad --target crypt_villus \
  --output-dir runs/demo/model --epochs 3 --device cpu

uv run --locked --extra cpu spatialtrace-graph predict \
  --input runs/demo/data.h5ad --checkpoint runs/demo/model/model.pt \
  --output runs/demo/predictions.csv --device cpu
```

## Use your data

Prepare an AnnData file with expression features, cell coordinates, section IDs, and annotations for the coordinates or regions you want to map. See the [input format](docs/input_formats.md) and [training guide](docs/retraining.md).

The paper uses intestinal tissue as an example. Training targets can be defined for your tissue.

Graph models are trained on your data. Pretrained image models are available in [SpatialTRACE-Image](https://github.com/amonell/SpatialTRACE-Image).

## More

- [Installation](docs/installation.md)
- [Model details](MODEL_CARD.md)
- [Reproduce the figures](reproduction/README.md)
- [Tests](VALIDATION.md)

[GPL-3.0-only](LICENSE).
