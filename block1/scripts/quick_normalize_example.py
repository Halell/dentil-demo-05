"""Quick script to run Block1 deterministic normalization + tokenization
on a single example line and print structured output.

Usage (from repo root, with venv active):
  python -m block1.scripts.quick_normalize_example
"""
from pprint import pprint

from block1.src.pipeline.n0_normalize import normalize_n0
from block1.src.pipeline.t1_tokenize import tokenize_t1

EXAMPLE_LINE = "מולטיוניט שתל14 18/0"

def run_example(line: str):
    n0 = normalize_n0(line)
    t1 = tokenize_t1(n0)
    # Build a concise dict result
    result = {
        "raw_text": n0.raw_text,
        "normalized_text": n0.normalized_text,
        "notes": n0.notes,
        "numbers": n0.numbers,
        "pairs": [p.model_dump() for p in n0.pairs],
        "tokens": [
            {
                "idx": tok.idx,
                "text": tok.text,
                "kind": tok.kind,
                "span": tok.span,
                "script": tok.script,
                "meta": tok.meta.model_dump() if tok.meta else None,
            }
            for tok in t1.tokens
        ],
    }
    return result

if __name__ == "__main__":
    out = run_example(EXAMPLE_LINE)
    pprint(out, width=120, compact=False)
