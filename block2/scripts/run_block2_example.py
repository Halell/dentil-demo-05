from __future__ import annotations
import os, json
from pathlib import Path

from block2.src.pipeline.router_block2 import run_block2_all

def main():
    # Expect merged_block1.jsonl already produced by Block1 CLI
    merged = Path('block1') / 'artifacts' / 'merged_block1.jsonl'
    if not merged.exists():
        raise SystemExit("Need block1/artifacts/merged_block1.jsonl. Run Block1 pipeline first.")
    out_dir = Path('block2') / 'artifacts'
    out_dir.mkdir(parents=True, exist_ok=True)
    res = run_block2_all(str(merged), str(out_dir))
    print('Block2 outputs:')
    for k,v in res.items():
        print(f"  {k}: {v}")

if __name__ == '__main__':
    main()
