# TissueMapper-Graph model card

TissueMapper-Graph supplies GPL-3.0-only training and inference code, without pretrained graph weights. Two coordinate regressors map expression features and within-section spatial neighborhoods to normalized [0, 1] outputs. Each uses two GATConv layers, ELU, dropout and sigmoid output. The separately trained BinaryGAT Peyer classifier produces a scalar logit. Input dimensionality and architecture settings are saved in each user's checkpoint; classification also stores training-section feature normalization and the prespecified decision threshold.

The paper used 30-dimensional scVI features, 64 hidden features, four first-layer attention heads and dropout 0.1 for coordinate regression. Its separate classifier used 128 hidden features, four heads, dropout 0.15 and a frozen threshold of 0.95. Dense visualization maps come from all-label fits and are not held-out performance estimates. Manuscript graph evaluation uses eight section-disjoint folds; the jointly learned scVI representation remains a feature-lineage limitation. Those frozen results are retained for figure reproduction, without distributing the graph checkpoint binaries.

## Graph identity

The historical paper-v1 coordinate graph construction selected non-self neighbor ranks 2–21: the original sklearn call already excluded self, then removed its first returned neighbor. Compatibility code preserves that recorded policy for previously saved checkpoints. Sections with 21 or fewer cells are rejected under this policy.

New training uses ordinary nearest-neighbor construction (knn), symmetrized edges and GATConv self loops. The historical paper-peyer-v1 policy matches the original classifier's cKDTree query and symmetrization. Checkpoints record their policy; inference never silently changes it. These distinctions do not revise the frozen paper results.

## Intended use and limitations

Research mapping of intestinal tissue, with study-specific training and anatomical validation. Save the fitted feature encoder used for training and reuse it for inference. Independently fitting another scVI encoder changes the feature basis even at the same dimensionality. The paper's original encoder is not needed for the supported workflow of training your own models.

Spatial neighbors affect predictions, so dropping unannotated surrounding cells changes the result. Gates, probability calibration and performance can shift with specimen preparation, expression panels and tissue domain. This is not a clinical diagnostic device.

Training and inference write input/output hashes and configuration in provenance JSON. Models can be saved and reused locally. The absence of downloadable graph checkpoints is an intentional release policy, not a missing dependency.
