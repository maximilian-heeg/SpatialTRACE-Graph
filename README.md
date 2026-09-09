# TissueMapper-Graph

Predict crypt–villus position, epithelial distance, and Peyer’s patch probability from spatial transcriptomics. Train models on your own scVI features and anatomical annotations.

## Install

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
gh repo clone amonell/TissueMapper-Graph
cd TissueMapper-Graph
uv sync --locked --extra cpu
```

For an NVIDIA GPU on Linux, use `--extra cu126` instead of `--extra cpu`.

## Try it

Create a small synthetic dataset, train a coordinate model, and predict:

```bash
uv run --locked --extra cpu tissuemapper-graph create-demo \
  --output runs/demo/data.h5ad

uv run --locked --extra cpu tissuemapper-graph train \
  --input runs/demo/data.h5ad --target crypt_villus \
  --output-dir runs/demo/model --epochs 3 --device cpu

uv run --locked --extra cpu tissuemapper-graph predict \
  --input runs/demo/data.h5ad --checkpoint runs/demo/model/model.pt \
  --output runs/demo/predictions.csv --device cpu
```

## Use your data

Prepare an AnnData file with expression features, cell coordinates, section IDs, and annotations. See the [input format](docs/input_formats.md) and [training guide](docs/retraining.md).

Graph models are trained on your data. Pretrained image models are available in [TissueMapper-Image](https://github.com/amonell/TissueMapper-Image).

## More

- [Installation](docs/installation.md)
- [Model details](MODEL_CARD.md)
- [Reproduce the figures](reproduction/README.md)
- [Tests](VALIDATION.md)

[GPL-3.0-only](LICENSE).
