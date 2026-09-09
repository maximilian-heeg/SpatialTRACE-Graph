# Example

The [README](../README.md#try-it) trains a coordinate model on synthetic data and predicts on the same AnnData file.

To train the other coordinate, change `--target crypt_villus` to `--target epithelial_distance` and choose a new output directory. The [training guide](../docs/retraining.md) also covers Peyer’s patch classification.

For real data, follow the [input format](../docs/input_formats.md). Keep unannotated neighboring cells in the input.
