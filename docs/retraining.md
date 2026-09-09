# Training

Prepare the [input schema](input_formats.md), with normalized sparse annotations and NaN for unlabeled cells. Keep train, validation and test sections separate. The CLI rejects sections assigned to multiple splits.

```bash
uv run --locked --extra cpu tissuemapper-graph train --input my_data.h5ad --target crypt_villus --output-dir runs/my_crypt --epochs 100 --patience 20 --device cpu
uv run --locked --extra cpu tissuemapper-graph train --input my_data.h5ad --target epithelial_distance --output-dir runs/my_epithelial --epochs 100 --patience 20 --device cpu
```

Each coordinate is fitted separately using Adam and labeled-cell MSE. Minimum validation MSE selects the checkpoint; test labels do not affect selection. Exact two-hop minibatches preserve incoming neighbors. Unlabeled cells contribute features, not supervised loss.

For a separate Peyer's patch classifier, add a `peyer_label` annotation column (0/1, or soft probabilities; NaN for unlabeled cells):

```bash
uv run --locked --extra cpu tissuemapper-graph train --input my_data.h5ad --target peyer_label --task peyer --output-dir runs/my_peyer --hidden-features 128 --dropout 0.15 --epochs 100 --device cpu
```

Classifier training uses binary cross-entropy and selects minimum validation loss. Feature normalization is fitted on training-section cells only and saved in the checkpoint. The default hard-call threshold is 0.5; `--threshold` specifies a value before training, without consulting test labels. This is a new classifier, not a replacement for the frozen paper classifier or its fixed threshold. Prediction reports probabilities and calls. Training and prediction also write input/output hashes, settings, and package versions in provenance JSON.

For NVIDIA acceleration, select the cu126 extra and --device cuda. Increase --batch-size according to memory.

New training uses the nearest 20 non-self neighbors, symmetrized, with GATConv self loops. The frozen paper coordinate models preserve their original graph convention; see the model card in the repository root. Generic training is not a rerun of the manuscript's locked experiments.

The supported portable training entry point is `tissuemapper-graph train`. Older annotation-conversion helpers remain under `scripts/`; they are outside the tested training and inference workflow.
