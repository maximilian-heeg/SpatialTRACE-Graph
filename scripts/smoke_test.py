"""Exercise installed entry points in a new directory; never changes source data."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work-dir', required=True, type=Path)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    args = parser.parse_args()
    root = args.work_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    records = []
    def run(tokens):
        command = [sys.executable, '-m', "spatial_axis_gat.cli", *map(str, tokens)]
        start = time.monotonic()
        result = subprocess.run(command, cwd=root, capture_output=True, text=True)
        records.append(dict(command=command, seconds=time.monotonic()-start, exit_code=result.returncode))
        (root/f'step_{len(records):02}.log').write_text(result.stdout+result.stderr)
        (root/'acceptance.json').write_text(json.dumps(records, indent=2))
        print(f"step {len(records)}: exit={result.returncode}, {records[-1]['seconds']:.1f}s", flush=True)
        if result.returncode:
            raise RuntimeError(result.stdout+result.stderr)
    run(['--help'])
    run(['create-demo', '--output', 'demo.h5ad'])
    for target in ['crypt_villus', 'epithelial_distance']:
        run(['train', '--input', 'demo.h5ad', '--target', target, '--output-dir', target, '--epochs', '3', '--device', args.device])
        run(['predict', '--input', 'demo.h5ad', '--checkpoint', target+'/model.pt',
             '--output', target+'/predictions.csv', '--device', args.device])
    run(['train', '--input', 'demo.h5ad', '--target', 'peyer_label', '--task', 'peyer',
         '--output-dir', 'peyer', '--epochs', '3', '--device', args.device])
    run(['predict', '--input', 'demo.h5ad', '--checkpoint', 'peyer/model.pt',
         '--output', 'peyer/predictions.csv', '--device', args.device])
    import pandas as pd
    assert len(pd.read_csv(root/'crypt_villus/predictions.csv')) == 96
    print(root/'acceptance.json', flush=True)


if __name__ == '__main__':
    main()
