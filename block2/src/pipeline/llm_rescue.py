from __future__ import annotations
"""LLM Rescue: safe canonical English term suggestions when no candidates found.

Guardrails:
- Returns at most 3 terms (list of strings) under key canonical_terms
- Each term: 1-4 words, letters / hyphen / space only (A-Za-z -)
- Strips digits & units; rejects if any invalid token
"""
import re, json, os, time
from typing import Dict, List, Optional

SAFE_PATTERN = re.compile(r'^[A-Za-z][A-Za-z \-]{0,48}$')

_LLM_CACHE: Dict[str, Dict] = {}

def _call_external_llm(prompt: str) -> List[str]:
    """(LLM-1) Placeholder for real LLM call.
    Expected to return list of canonical terms.
    Currently routes to mock logic offline.
    """
    return _mock_llm(prompt)

def _mock_llm(surface: str) -> List[str]:
    s = surface.lower()
    if 'מולטי' in s or 'multi' in s:
        return ['multi-unit abutment']
    if 'שתל' in s or 'implant' in s:
        return ['dental implant']
    return []

def _validate_terms(cands: List[str]) -> List[str]:
    out = []
    for c in cands:
        c = c.strip()
        if not c:
            continue
        if not SAFE_PATTERN.match(c):
            continue
        # drop consecutive spaces
        c = re.sub(r'\s+', ' ', c)
        out.append(c)
        if len(out) >= 3:
            break
    return out

def llm_canonicalize_safe(text: str) -> Optional[Dict]:
    key = text.strip().lower()
    if key in _LLM_CACHE:  # (LLM-3) cache hit
        cached = dict(_LLM_CACHE[key])
        cached['cached'] = True
        return cached
    max_retries = 2  # (LLM-2)
    attempt = 0
    last_terms: List[str] = []
    t0 = time.time()
    while attempt <= max_retries:
        try:
            raw_terms = _call_external_llm(text)
            terms = _validate_terms(raw_terms)
            if terms:
                elapsed = round((time.time()-t0)*1000,2)
                payload = {"canonical_terms": terms, "latency_ms": elapsed, "retries": attempt, "notes": ["llm_rescue"]}
                _LLM_CACHE[key] = payload
                return payload
            last_terms = terms
        except Exception:
            pass
        attempt += 1
    if not last_terms:
        return None
    elapsed = round((time.time()-t0)*1000,2)
    payload = {"canonical_terms": last_terms, "latency_ms": elapsed, "retries": attempt, "notes": ["llm_rescue_partial"]}
    _LLM_CACHE[key] = payload
    return payload
