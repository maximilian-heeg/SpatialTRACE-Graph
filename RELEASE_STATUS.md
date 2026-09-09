# Release candidate status

The GPL-3.0-only release candidate includes uv-locked CPU/GPU installs, own-data coordinate/classifier training and inference, synthetic examples, and complete paper rendering code. The source is hosted in the private `amonell/TissueMapper-Graph` GitHub repository. Graph weights are not distributed. See `VALIDATION.md` for executed local tests and GitHub Actions for hosted checks. This is not a public release.

Before public release:

- Obtain authorization to change the repository from private to public.
- Upload the verified paper-input bundle after confirming permission to redistribute it.

The original development repository is preserved separately. History-free source exports omit Git metadata. Synthetic tutorials are not paper evaluations. The original paper scVI encoder is not a release requirement: users train their own feature encoder and graph models.
