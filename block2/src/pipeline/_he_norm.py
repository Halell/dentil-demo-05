from __future__ import annotations
from typing import List
try:
    from rapidfuzz.distance import DamerauLevenshtein as DL  # type: ignore
except Exception:  # fallback simple edit distance
    class _DL:
        @staticmethod
        def distance(a: str, b: str) -> int:
            # crude distance
            return abs(len(a) - len(b))
        @staticmethod
        def normalized_similarity(a: str, b: str) -> float:
            if not a and not b:
                return 1.0
            dist = _DL.distance(a,b)
            mx = max(len(a), len(b)) or 1
            return max(0.0, 1.0 - dist/mx)
    DL = _DL()  # type: ignore

STRIP_CHARS = {" ", "-", "\u05F4", "\u05F3"}  # space, hyphen, gershayim, geresh

def norm_he(s: str) -> str:
    s = s or ""
    out = []
    for ch in s:
        if ch in STRIP_CHARS:
            continue
        out.append(ch)
    return "".join(out)

def he_fuzzy_ok(q: str, cand: str, max_ed: int = 1, min_ratio: int = 90) -> bool:
    qn, cn = norm_he(q), norm_he(cand)
    try:
        if DL.distance(qn, cn) <= max_ed:
            return True
        sim = DL.normalized_similarity(qn, cn) * 100.0
        return sim >= min_ratio
    except Exception:
        return False
