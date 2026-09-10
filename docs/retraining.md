# Train and predict

Prepare the [AnnData input](input_formats.md). Assign each section to training, validation, or testing. Use NaN for unannotated cells.

## Coordinate models

Define a coordinate for your tissue and scale its annotations from 0 to 1. Pass its column name with `--target`. Train each coordinate separately; the commands below use the paper’s intestinal coordinates as examples:

```bash
uv run --locked --extra cpu tissuemapper-graph train \
  --input my_data.h5ad --target crypt_villus \
  --output-dir runs/crypt --epochs 100 --patience 20 --device cpu

uv run --locked --extra cpu tissuemapper-graph train \
  --input my_data.h5ad --target epithelial_distance \
  --output-dir runs/epithelial --epochs 100 --patience 20 --device cpu
```

Training uses Adam and mean squared error on annotated cells. The model with the lowest validation loss is saved as `model.pt`.

Predict with either saved model:

```bash
uv run --locked --extra cpu tissuemapper-graph predict \
  --input my_data.h5ad --checkpoint runs/crypt/model.pt \
  --output runs/crypt_predictions.csv --device cpu
```

## Region classification

Use binary labels to mark a region of interest. The example below classifies Peyer’s patches. For another region, replace `peyer_label` with your label column; `--task peyer` selects the binary classifier.

Add a `peyer_label` column with binary labels or probabilities from 0 to 1. Use NaN for unannotated cells.

```bash
uv run --locked --extra cpu tissuemapper-graph train \
  --input my_data.h5ad --target peyer_label --task peyer \
  --output-dir runs/peyer --hidden-features 128 --dropout 0.15 \
  --epochs 100 --device cpu
```

The classifier uses binary cross-entropy and selects the model with the lowest validation loss. Feature normalization is fitted on training sections and saved with the model. Predictions include probabilities and binary calls.

The default threshold is 0.5. Set a different threshold with `--threshold` before training.

## Neighbors and batches

New models use the nearest 20 non-self neighbors within each section. Edges are symmetrized, and the attention layers add self-loops. Unannotated cells contribute features to their neighbors.

Minibatches retain each target cell’s two-hop neighborhood. Adjust `--batch-size` to fit memory. For an NVIDIA GPU, use `--extra cu126` and `--device cuda`.

See the [model card](https://github.com/amonell/TissueMapper-Graph/blob/main/MODEL_CARD.md) for the paper’s graph settings and evaluation design.
