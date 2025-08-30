from pathlib import Path
import sys
ROOT = Path(__file__).parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from block2.src.pipeline.cand_faiss import _load_index

def main():
    base = Path(__file__).parents[1]
    idx, meta, model = _load_index(base)
    if model is None:
        print('Model not loaded')
        return
    out = model.encode(['warmup embedding run'])
    print('Prewarm ok shape', getattr(out,'shape', None))

if __name__ == '__main__':
    main()
