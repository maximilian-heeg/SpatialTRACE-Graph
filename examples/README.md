# Examples

Run `tissuemapper-graph create-demo --output runs/demo/data.h5ad` to generate a complete synthetic AnnData tutorial with three section-disjoint partitions. The README gives tested commands for training both coordinates and inference; the training guide also covers the separate classifier.

For real data, use the schema in `docs/input_formats.md`. Keep all available within-section neighboring cells, including cells without annotations. No paper data or pretrained scVI reference encoder is embedded in the example.
