"""SpatialTRACE-Graph command line."""
import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(prog="spatialtrace-graph")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("create-demo", help="Write a self-contained synthetic tutorial dataset")
    demo.add_argument("--output", type=Path, required=True)
    for command in ["train", "predict"]:
        p = sub.add_parser(command)
        p.add_argument("--input", type=Path, required=True)
        p.add_argument("--feature-key", default="X_scVI")
        p.add_argument("--spatial-key", default="X_spatial")
        p.add_argument("--section-key", default="section_id")
        p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
        p.add_argument("--batch-size", type=int, default=128)
        p.add_argument("--threads", type=int, default=4)
        if command == "predict":
            p.add_argument("--checkpoint", type=Path, required=True)
            p.add_argument("--output", type=Path, required=True)
        else:
            p.add_argument("--target", required=True)
            p.add_argument("--task", choices=["coordinate", "peyer"], default="coordinate")
            p.add_argument("--threshold", type=float, default=.5, help="Prespecified probability threshold for a newly trained classifier")
            p.add_argument("--split-key", default="split")
            p.add_argument("--output-dir", type=Path, required=True)
            p.add_argument("--epochs", type=int, default=100)
            p.add_argument("--patience", type=int, default=20)
            p.add_argument("--neighbors", type=int, default=20)
            p.add_argument("--hidden-features", type=int, default=64)
            p.add_argument("--heads", type=int, default=4)
            p.add_argument("--dropout", type=float, default=.1)
            p.add_argument("--lr", type=float, default=.001)
            p.add_argument("--weight-decay", type=float, default=.00001)
            p.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)
    if args.command == "create-demo":
        from .workflow import create_demo
        create_demo(args.output)
    else:
        import torch
        if args.threads < 1:
            parser.error("--threads must be positive")
        torch.set_num_threads(args.threads)
        if args.device == "cuda" and not torch.cuda.is_available():
            parser.error("CUDA was requested but is unavailable; install the cu126 extra or choose --device cpu")
        from . import workflow
        getattr(workflow, args.command)(args)


if __name__ == "__main__":
    main()
