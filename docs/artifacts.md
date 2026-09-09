# Models and outputs

TissueMapper-Graph is a GPL-3.0-only source release. It supplies training and inference code without pretrained graph weights or a paper-specific scVI encoder. Pretrained downloads belong to TissueMapper-Image.

Graph training writes `model.pt`, containing validation-selected weights, architecture, feature-space identity and graph construction settings, plus history, split, metrics and provenance files. Keep these with your fitted scVI encoder and gene/preprocessing metadata. Use that saved checkpoint with `tissuemapper-graph predict` on compatible features. The loader uses tensor-safe deserialization; only load checkpoints from trusted sources.

Paper figures render from separately distributed frozen predictions, metrics and microscopy inputs. Graph checkpoint binaries are omitted; their recorded identities are retained as provenance. Rendering does not refit models or read omitted graph weights.
