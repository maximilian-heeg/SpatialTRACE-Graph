# Input format

Use an AnnData `.h5ad` file with unique cell IDs in `obs_names`.

| Field | Contents |
| --- | --- |
| `obsm["X_scVI"]` | Dense cell × feature matrix with finite values |
| `obsm["X_spatial"]` | Cell × 2 coordinates, ordered x, y |
| `obs["section_id"]` | Section ID |
| `uns["tissuemapper_feature_space"]` | Name of the fitted feature encoder |
| `obs["split"]` | `train`, `validation`, or `test`; one split per section |
| `obs["crypt_villus"]` | Labels from 0 to 1; NaN for unannotated cells |
| `obs["epithelial_distance"]` | Labels from 0 to 1; NaN for unannotated cells |

The split and label fields are required for training. For Peyer’s patch classification, add `obs["peyer_label"]` with labels from 0 to 1.

Keep spatial units consistent within each section. Include surrounding unannotated cells so the model can use their features.

Give each fitted encoder a name, such as `my-study-scvi-v1`. Save the encoder, gene order, and preprocessing settings with the model. Reuse them when predicting on new cells: a newly fitted scVI encoder produces a different feature space.

Use `tissuemapper-graph train --help` to change the default column names.
