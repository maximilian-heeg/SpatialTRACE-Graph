# Input format

Use an AnnData h5ad with unique cell IDs in obs_names.

| Field | Contents |
| --- | --- |
| obsm["X_scVI"] | Finite dense cells × features matrix |
| obsm["X_spatial"] | Cells × 2 spatial coordinates, x then y |
| obs["section_id"] | Section identifier |
| uns["tissuemapper_feature_space"] | Identifier of the fitted feature encoder |
| obs["split"] | For training: train, validation, or test; one split per section |
| obs["crypt_villus"], obs["epithelial_distance"] | Normalized labels in [0, 1]; NaN for unannotated cells |

CLI flags support alternative feature, spatial, section and split keys. Coordinate units must be consistent within each section. Include surrounding unlabeled cells: removing them changes the graph.

For new training, give your embedding a new identifier, such as my-study-scvi-v1, and save its fitted encoder, gene order and preprocessing. Reuse that encoder for deployment. Metadata validation cannot prove that an identifier was assigned honestly.

Graph weights are trained on your own annotations and feature basis. No pretrained graph weights or paper scVI encoder are distributed. Reuse the fitted encoder associated with your saved checkpoint; matching the number of features alone does not ensure compatibility.

The tutorial uses the field name X_scVI for interface testing but contains explicitly labeled synthetic features.
