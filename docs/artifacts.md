# Saved models and outputs

Training saves:

- `model.pt`: the model selected by validation loss, with its architecture and input settings.
- `history.csv`: training and validation losses.
- `split.csv`: training, validation, and test assignments.
- Metrics and provenance files: performance, settings, input hashes, and package versions.

Keep these files with your fitted scVI encoder and preprocessing settings. Use `spatialtrace-graph predict` to apply the saved model to compatible features.

Graph weights are trained on your own annotations. The figure workflow uses saved paper predictions and metrics supplied separately.
