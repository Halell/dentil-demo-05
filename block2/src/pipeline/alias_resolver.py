from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, Dict, Any, List

import threading

_lex_cache = None
_lex_lock = threading.Lock()

def _load_lexicon(base: Path):
    global _lex_cache
    if _lex_cache is not None:
        return _lex_cache
    lex_path = base / 'artifacts' / 'lexicon' / 'ohd_lexicon.jsonl'
    items = []
    if not lex_path.exists():
        _lex_cache = items
        return items
    with open(lex_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                iri = rec.get('iri')
                label = rec.get('label')
                if iri and label:
                    syns = rec.get('synonyms', []) or []
                    items.append({'iri': iri, 'label': label, 'synonyms': syns})
            except Exception:
                continue
    _lex_cache = items
    return items

def _vector_top_match(text: str, base: Path) -> Optional[Dict[str, Any]]:
    """Optional vector similarity lookup (Top-1) using existing FAISS index if present.
    Returns candidate dict or None."""
    try:
        import faiss  # type: ignore
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception:
        return None
    idx_path = base / 'artifacts' / 'vectors' / 'ohd.faiss'
    meta_path = base / 'artifacts' / 'vectors' / 'ohd_meta.json'
    if not idx_path.exists() or not meta_path.exists():
        return None
    try:
        index = faiss.read_index(str(idx_path))
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        rows = meta.get('rows') or meta.get('entries') or []
        if not rows:
            return None
        model_name = meta.get('model_name') or 'sentence-transformers/all-MiniLM-L6-v2'
        model = SentenceTransformer(model_name)
        emb = model.encode([text])
        import numpy as np
        D, I = index.search(emb.astype('float32'), 3)
        if I.size == 0:
            return None
        best_idx = I[0][0]
        if best_idx < 0:
            return None
        row = rows[best_idx]
        return {'iri': row['iri'], 'label': row['label'], 'source': 'vector'}
    except Exception:
        return None

def resolve_alias_to_ohd(alias: str) -> Optional[Dict[str, Any]]:
    """Resolve free-text English alias to OHD IRI with staged strategy:
    1. Exact label match
    2. Exact synonym match
    3. Containment (alias contained in label/synonym or vice versa)
    4. Vector top-3 then lexical proximity (shortest edit distance) re-rank
    Returns dict {iri,label,iri_source} or None."""
    alias_norm = (alias or '').strip()
    if not alias_norm:
        return None
    alias_l = alias_norm.lower()
    base = Path(__file__).parents[3]
    with _lex_lock:
        lex = _load_lexicon(base)
    if not lex:
        return None
    # 1 exact label
    for rec in lex:
        if rec['label'].lower() == alias_l:
            return {'iri': rec['iri'], 'label': rec['label'], 'iri_source': 'ohd_label'}
    # 2 exact synonym
    for rec in lex:
        for syn in rec['synonyms']:
            if syn.lower() == alias_l:
                return {'iri': rec['iri'], 'label': rec['label'], 'iri_source': 'ohd_synonym'}
    # 3 containment
    cont = []
    for rec in lex:
        lbl_l = rec['label'].lower()
        if alias_l in lbl_l or lbl_l in alias_l:
            cont.append(rec)
        else:
            for syn in rec['synonyms']:
                syn_l = syn.lower()
                if alias_l in syn_l or syn_l in alias_l:
                    cont.append(rec)
                    break
    if cont:
        # pick shortest label length tie-broken by earliest
        best = sorted(cont, key=lambda r: len(r['label']))[0]
        return {'iri': best['iri'], 'label': best['label'], 'iri_source': 'resolved_alias'}
    # 4 vector top3 + lexical proximity
    try:
        import faiss  # type: ignore
        from sentence_transformers import SentenceTransformer  # type: ignore
        idx_path = base / 'artifacts' / 'vectors' / 'ohd.faiss'
        meta_path = base / 'artifacts' / 'vectors' / 'ohd_meta.json'
        if idx_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
            rows = meta.get('rows') or meta.get('entries') or []
            if rows:
                model_name = meta.get('model_name') or 'sentence-transformers/all-MiniLM-L6-v2'
                model = SentenceTransformer(model_name)
                index = faiss.read_index(str(idx_path))
                import numpy as np
                emb = model.encode([alias_norm])
                D, I = index.search(emb.astype('float32'), 3)
                cands: List[Dict[str, Any]] = []
                for rank, idx in enumerate(I[0]):
                    if idx < 0 or idx >= len(rows):
                        continue
                    row = rows[idx]
                    cands.append(row)
                # simple edit distance scoring
                def _ed(a,b):
                    try:
                        from rapidfuzz.distance import Levenshtein
                        return Levenshtein.distance(a.lower(), b.lower())
                    except Exception:
                        return abs(len(a)-len(b))
                best_row = None
                best_score = 1e9
                for r in cands:
                    score = min(_ed(alias_norm, r['label']), *[_ed(alias_norm, s) for s in r.get('synonyms',[])[:3]] or [1e9])
                    if score < best_score:
                        best_score = score
                        best_row = r
                if best_row:
                    return {'iri': best_row['iri'], 'label': best_row['label'], 'iri_source': 'resolved_alias'}
    except Exception:
        pass
    return None
