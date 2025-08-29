from __future__ import annotations
import re

HEBREW_LETTERS = set(chr(c) for c in range(0x0590, 0x05FF))

_SPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[\u200f\u200e\ufeff]")  # remove bidi / BOM artifacts

FINAL_MAP = {
    'ך': 'כ',
    'ם': 'מ',
    'ן': 'נ',
    'ף': 'פ',
    'ץ': 'צ'
}

def normalize_hebrew(text: str) -> str:
    """Lightweight Hebrew normalization used for gazetteer & fuzzy matching.
    - Strip BOM / bidi marks
    - Collapse whitespace
    - Map final letters to base form
    - Lowercase (safe for Latin parts)
    """
    if not text:
        return text
    t = _PUNCT_RE.sub('', text)
    t = ''.join(FINAL_MAP.get(ch, ch) for ch in t)
    t = _SPACE_RE.sub(' ', t).strip()
    return t.lower()

def is_mostly_hebrew(text: str) -> bool:
    if not text:
        return False
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return False
    heb = sum(1 for ch in letters if ch in HEBREW_LETTERS)
    return heb / max(1, len(letters)) >= 0.5
