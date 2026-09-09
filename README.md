# TissueMapper-Graph

Predict normalized crypt–villus position and epithelial distance from expression features and within-section spatial neighborhoods. The two coordinates use separate graph attention regressors. A separate graph classifier predicts Peyer’s patch probability.

## Install with uv

Clone the repository first (GitHub access is required while it is private):

```bash
gh repo clone amonell/TissueMapper-Graph
cd TissueMapper-Graph
```

From this checkout, uv installs Python 3.12 and the locked dependencies:

```bash
uv sync --locked --extra cpu --extra dev
uv run --locked --extra cpu tissuemapper-graph --help
```

For Linux with an NVIDIA GPU, replace `--extra cpu` with `--extra cu126` in installation and subsequent commands. Do not select both extras. No compiled PyG sampling extension is required.

## Self-contained example

These commands generate synthetic data, train with section-disjoint validation, and run the selected model. They test software execution, not paper accuracy. Use a new output directory for each run.

```bash
uv run --locked --extra cpu tissuemapper-graph create-demo --output runs/demo/data.h5ad
uv run --locked --extra cpu tissuemapper-graph train --input runs/demo/data.h5ad --target crypt_villus --output-dir runs/demo/crypt --epochs 3 --device cpu
uv run --locked --extra cpu tissuemapper-graph train --input runs/demo/data.h5ad --target epithelial_distance --output-dir runs/demo/epithelial --epochs 3 --device cpu
uv run --locked --extra cpu tissuemapper-graph predict --input runs/demo/data.h5ad --checkpoint runs/demo/crypt/model.pt --output runs/demo/predictions.csv --device cpu
```

Training writes the validation-selected `model.pt`, `history.csv`, `split.csv`, and partition-level metrics. Inference preserves cell IDs.

## Train on your own data

TissueMapper-Graph provides training and inference code without pretrained graph weights. Fit scVI features for your data, add anatomical annotations and section splits, then use `train` and `predict` as above. The [input schema](docs/input_formats.md) and [training guide](docs/retraining.md) describe both coordinate regressors and the separate Peyer's patch classifier.

Save your fitted feature encoder together with your graph checkpoints. Inference requires the same feature basis used for training; independently refitting scVI changes that basis. The paper's original scVI encoder is not required for training your own models. Pretrained model downloads are provided only by TissueMapper-Image.

## Documentation and tests

- [Installation](docs/installation.md)
- [Input schema](docs/input_formats.md)
- [Training](docs/retraining.md)
- [Model card](MODEL_CARD.md)
- [Artifacts](docs/artifacts.md)
- [Paper figures](reproduction/README.md)
- [Release status](RELEASE_STATUS.md)
- [Executed validation](VALIDATION.md)

```bash
uv run --locked --extra cpu --extra dev pytest -q
```

The Python import `spatial_axis_gat` is retained for compatibility. The repository, distribution and CLI use TissueMapper-Graph. Code is licensed under GPL-3.0-only; see [LICENSE](LICENSE). Figure-data terms are separate.
