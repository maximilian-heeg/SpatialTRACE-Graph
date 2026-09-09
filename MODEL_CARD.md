# TissueMapper-Graph

## Model

Two graph attention regressors predict crypt–villus position and epithelial distance from expression features and spatial neighbors. Outputs range from 0 to 1.

Each regressor has two GATConv layers with ELU, dropout, and a sigmoid output. A separate BinaryGAT classifier predicts Peyer’s patch probability.

The paper used 30 scVI features, 64 hidden features, four first-layer attention heads, and dropout of 0.1 for coordinate models. The classifier used 128 hidden features, four heads, dropout of 0.15, and a threshold of 0.95.

## Spatial graph

New models use the nearest 20 non-self neighbors within each section. Edges are symmetrized, and GATConv adds self-loops.

Historical checkpoints record one of two older policies:

- `paper-v1`: coordinate models used neighbor ranks 2–21. The original code removed the first neighbor after self had already been excluded. This policy requires more than 21 cells per section.
- `paper-peyer-v1`: the classifier used a cKDTree query followed by edge symmetrization.

Inference uses the policy recorded in the checkpoint.

## Use on new data

Train on your own expression features and annotations. Save the fitted scVI encoder, gene order, and preprocessing settings with the model. Reuse them for prediction.

Include unannotated neighboring cells. Their features affect predictions.

Validate anatomical gates and classifier calibration in each study. Tissue preparation, gene panels, and anatomy can affect performance. This software is intended for research.

## Paper evaluation

Coordinate performance was evaluated across eight section-disjoint folds. scVI features were learned jointly across sections, which limits the independence of that evaluation. Dense maps use models fitted on all annotations and serve as visualizations.

The figure bundle contains saved predictions and metrics. Graph checkpoint files are excluded from the release.
