# Release validation

Validated locally on Linux, Python 3.12, PyTorch 2.12.1, CPU and NVIDIA RTX A6000/CUDA 12.6. Wheels were installed into separate uv environments without editable source installs, and commands ran in fresh output directories. No publication or GitHub-hosted CI run is claimed.

- All 10 source-only release unit tests passed. The 8-command acceptance workflow passed on CPU and GPU, covering synthetic AnnData generation, separate coordinate training/inference, and separate classifier training/inference.
- Exact two-hop minibatch inference agrees with whole-graph inference in the test graph. Tests reject section overlap, incompatible feature-space identities and overwrites, and ensure pretrained-download commands are not offered.
- The newly trained classifier saves training-section-only feature normalization; test labels do not select the checkpoint.
- Previously staged Graph artifacts matched their frozen originals; those exports are retained privately and are not part of this release.

On 32 prespecified cells in a real reference section, maximum absolute differences from the correct frozen outputs were 1.2 × 10⁻⁷ for crypt–villus position, 8.9 × 10⁻⁸ for epithelial distance and 6.3 × 10⁻⁸ for Peyer's patch probability. These are bounded compatibility checks, not new performance estimates. The coordinate and classifier source tables have different global row indices; matching used stable cell identities within the same section.

All eight paper figures were redrawn from relocated frozen inputs and matched the current full-page layouts pixel-for-pixel in a 72-dpi raster comparison. The figure runner additionally checks panel recomposition at 150 dpi, fonts, text, geometry and raster contracts.

```bash
python scripts/smoke_test.py --work-dir /new/temporary/graph-run --device cpu
```

Use a CUDA-installed interpreter and `--device cuda` for GPU acceptance. Reports and logs are written inside the work directory. The current workflow trains and reloads users' own checkpoints and requires no released graph weights or original paper scVI encoder. The real-reference checks above document historical internal compatibility, not downloadable pretrained models.
