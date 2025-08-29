"""Run a single ad-hoc line through Block 1 + Block 2 full pipelines.

Input hardcoded: "מולטיוניט שלת14 18/0 " (intentional variant / typo to observe behavior)

Outputs printed to stdout (compact) and written to temp artifacts under ./artifacts/ad_hoc_demo/
"""
from __future__ import annotations
import json, tempfile, shutil
from pathlib import Path

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[0].parent
sys.path.insert(0, str(PROJECT_ROOT))
from block1.src.pipeline.ops_apply import orchestrate_block1  # type: ignore
from block2.src.pipeline.router_block2 import run_block2_all  # type: ignore

LINE = "מולטיוניט שלת14 18/0 "  # note: typo 'שלת' -> expected 'שתל'

def main():
    base = Path('artifacts') / 'ad_hoc_demo'
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True, exist_ok=True)

    # ---- Block 1 ----
    b1 = orchestrate_block1(LINE)
    b1_path = base / 'merged_block1.jsonl'
    with open(b1_path, 'w', encoding='utf-8') as f:
        merged_line = {
            'raw_text': LINE.strip(),
            'n0': {
                'raw_text': b1['normalized']['original'],
                'normalized_text': b1['normalized']['text'],
                'notes': b1['normalized']['notes']
            },
            't1': {
                'text': b1['normalized']['applied_text'],
                'tokens': b1['tokens']
            },
            'llm_aug': b1.get('augment') or None
        }
        f.write(json.dumps(merged_line, ensure_ascii=False)+'\n')

    print('--- Block 1 Output ---')
    print(json.dumps(merged_line, ensure_ascii=False, indent=2))

    # ---- Block 2 ----
    b2_outputs = run_block2_all(str(b1_path), str(base))

    # Read merged candidates & mv
    merged_candidates = []
    mv_candidates = []
    merged_file = Path(b2_outputs['merged'])
    if merged_file.exists():
        with open(merged_file, 'r', encoding='utf-8') as f:
            merged_candidates = [json.loads(l) for l in f]
    mv_file = Path(b2_outputs['mv'])
    if mv_file.exists():
        with open(mv_file, 'r', encoding='utf-8') as f:
            mv_candidates = [json.loads(l) for l in f]

    print('\n--- Block 2 Candidates (merged) ---')
    print(json.dumps(merged_candidates, ensure_ascii=False, indent=2))
    print('\n--- Block 2 MV (v2) ---')
    print(json.dumps(mv_candidates, ensure_ascii=False, indent=2))
    print(f"\nArtifacts written under: {base}")

if __name__ == '__main__':
    main()
