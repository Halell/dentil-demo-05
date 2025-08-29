from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from ..common.config import CONFIG
from ..common.schemas import VectorHit

_model_cache = None
_index_cache = None
_meta_cache = None
_he2en = None


def _load_index(base: Path):
    global _index_cache, _meta_cache, _model_cache
    if _index_cache is None:
        idx_path = base / 'artifacts' / 'vectors' / 'ohd.faiss'
        meta_path = base / 'artifacts' / 'vectors' / 'ohd_meta.json'
        if not idx_path.exists() or not meta_path.exists():
            return None, None, None
        try:
            _index_cache = faiss.read_index(str(idx_path))
            with open(meta_path, 'r', encoding='utf-8') as f:
                raw = f.read().strip()
                _meta_cache = json.loads(raw) if raw else {}
        except Exception:
            _index_cache = None
            _meta_cache = None
            return None, None, None
        model_dir = (_meta_cache or {}).get('model_name') or (_meta_cache or {}).get('model_dir') or 'sentence-transformers/all-MiniLM-L6-v2'
        try:
            _model_cache = SentenceTransformer(model_dir)
        except Exception:
            _model_cache = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    # load HE->EN static mapping
    global _he2en
    if _he2en is None:
        map_path = base / 'block2' / 'data' / 'dictionaries' / 'he2en_static.json'
        if map_path.exists():
            try:
                _he2en = json.loads(map_path.read_text(encoding='utf-8'))
            except Exception:
                _he2en = {}
        else:
            _he2en = {}
    return _index_cache, _meta_cache, _model_cache


def build_query_text(surface: str, neighbor_tokens: List[str], canonical_terms: List[str] | None) -> str:
    if canonical_terms:
        return ' '.join(canonical_terms[:3])
    # he->en fallback
    if _he2en and surface in _he2en:
        return _he2en[surface][0]
    ctx = ' '.join(neighbor_tokens[:2]) if neighbor_tokens else ''
    return f"{surface} {ctx}".strip()


def vector_candidates_faiss(line: Dict, topk: int) -> List[VectorHit]:
    base = Path(__file__).parents[3]
    idx, meta, model = _load_index(base)
    if idx is None or meta is None or model is None or not meta.get('rows'):
        return []
    n0 = line.get('n0', {})
    t1 = line.get('t1', {})
    tokens = t1.get('tokens', [])
    canonical_terms = (line.get('llm_aug') or {}).get('canonical_terms') or []

    hits: List[VectorHit] = []
    for tok in tokens:
        if tok.get('kind') in {'number','pair','unit'}:
            continue
        surface = tok['text']
        # gather neighbor tokens (simple left/right within 2)
        idx_tok = tok['idx']
        neigh = [t['text'] for t in tokens if abs(t['idx']-idx_tok) <= 2 and t['idx'] != idx_tok]
        qtext = build_query_text(surface, neigh, canonical_terms)
        qemb = model.encode([qtext])
        D, I = idx.search(qemb.astype(np.float32), topk)
        cand = []
        for score, ii in zip(D[0], I[0]):
            meta_row = meta['rows'][ii]
            cand.append({
                'iri': meta_row['iri'],
                'label': meta_row['label'],
                'score_vec': float(score)
            })
        vh = VectorHit(
            mention_id=f"v_{tok['idx']}",
            surface=surface,
            context=qtext,
            span=tuple(tok.get('span', (0,0))),
            candidates_vec=cand
        )
        hits.append(vh)
    return hits


def run_vector_faiss(merged_block1_path: str, out_path: str) -> None:
    with open(merged_block1_path, 'r', encoding='utf-8') as fin, open(out_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            rec = json.loads(line)
            hits = vector_candidates_faiss(rec, CONFIG.TOPK_VEC)
            for h in hits:
                fout.write(h.model_dump_json()+'\n')
