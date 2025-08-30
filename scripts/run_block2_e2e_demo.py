import json, time, tempfile
from pathlib import Path
import sys
ROOT = Path(__file__).parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from block2.src.pipeline.router_block2 import run_block2_all
from scripts.prewarm_embeddings import main as prewarm

SAMPLES = [
    "שתל",
    "מולטיוניט",
    "שלת",
    "abutment titanium",
    "12345",  # should be filtered
]

def tokens_for(text: str):
    toks = []
    pos = 0
    for idx, part in enumerate(text.split()):
        span = (pos, pos+len(part))
        toks.append({"idx": idx, "text": part, "span": [span[0], span[1]], "kind": "word"})
        pos += len(part) + 1
    return toks

def build_block1_file(lines):
    p = Path(tempfile.mkdtemp()) / 'block1_input.jsonl'
    with open(p,'w',encoding='utf-8') as f:
        for i, text in enumerate(lines):
            rec = {"id": i, "raw": text, "t1": {"tokens": tokens_for(text)}}
            f.write(json.dumps(rec, ensure_ascii=False)+'\n')
    return p

def main():
    prewarm()
    block1 = build_block1_file(SAMPLES)
    out_dir = Path('block2_output_demo')
    t0 = time.time()
    res = run_block2_all(str(block1), str(out_dir))
    dt = time.time()-t0
    print('Run complete in %.2fs' % dt)
    print('Outputs:', res)
    merged = Path(res['merged'])
    print('\nMerged candidates (first lines):')
    with open(merged,'r',encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i>=10: break
            print(line.strip())

if __name__ == '__main__':
    main()
