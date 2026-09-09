# Tests

Tested on Linux with Python 3.12, PyTorch 2.12.1, CPU, and an NVIDIA RTX A6000 with CUDA 12.6. Packages were installed as wheels in separate uv environments.

## Software checks

- 10 unit tests passed.
- The eight-command training and prediction workflow passed on CPU and GPU.
- Minibatch predictions matched whole-graph predictions in the test graph.
- Tests covered section splits, feature compatibility, classifier normalization, and file protection.
- GitHub Actions passed installation, unit tests, wheel building, and the example workflow.

Run the unit tests:

```bash
uv run --locked --extra cpu --extra dev pytest -q
```

Run the training and prediction checks:

```bash
uv run --locked --extra cpu python scripts/smoke_test.py \
  --work-dir runs/graph_checks --device cpu
```

Use a new work directory. For GPU checks, use `--extra cu126` and `--device cuda`. Logs are saved in the work directory.

## Comparison with the paper code

On 32 cells from a reference section, maximum prediction differences were 1.2 × 10⁻⁷ for crypt–villus position, 8.9 × 10⁻⁸ for epithelial distance, and 6.3 × 10⁻⁸ for Peyer’s patch probability. This checks implementation agreement rather than biological accuracy.

All eight figures matched the current page layouts pixel-for-pixel at 150 dpi. Figure checks also covered fonts, text, panel placement, and image resolution.

The machine-readable test record is in `validation/acceptance.json`.
