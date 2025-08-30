from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Deque, Tuple
from collections import deque
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
try:
    import torch
except Exception:  # torch optional
    torch = None
from ..common.config import CONFIG
import os, sys
from ..common.schemas import VectorHit
import time

_model_cache = None
_index_cache = None
_meta_cache = None
_he2en = None
_perf_stats = {
    'encode_times': [],
    'cache_hits': 0,
    'cache_misses': 0,
    'batch_sizes': []
}

# Simple LRU cache for surface->embedding (VEC-2)
_embed_cache: Dict[str, np.ndarray] = {}
_embed_lru: Deque[str] = deque()
_EMBED_CACHE_MAX = int(os.getenv('EMBED_CACHE_MAX', '256'))


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
        # (VEC-3) allow override to offline packaged model under artifacts/models/<name>
        packaged_dir = base / 'artifacts' / 'models'
        env_model = os.getenv('EMBED_MODEL_LOCAL')
        if env_model:
            candidate_local = packaged_dir / env_model
            if candidate_local.exists():
                model_dir = str(candidate_local)
        try:
            device = 'cuda' if (torch is not None and torch.cuda.is_available()) else 'cpu'
            _model_cache = SentenceTransformer(model_dir, device=device)
        except Exception:
            _model_cache = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        # warm-up single encode to mitigate first-call latency (especially on GPU graph init)
        try:
            _ = _model_cache.encode(["warmup"], show_progress_bar=False)
        except Exception:
            pass
        # C2 health log
        try:
            ntotal = _index_cache.ntotal if _index_cache is not None else 0
            if ntotal == 0:
                print('[vector] WARNING empty FAISS index (ntotal=0)', file=sys.stderr)
            else:
                print(f'[vector] FAISS index loaded ntotal={ntotal}', file=sys.stderr)
        except Exception:
            pass
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
    # Priority: canonical provided > he->en mapping > enriched surface+neighbors
    if canonical_terms:
        return ' '.join(canonical_terms[:3])
    if _he2en and surface in _he2en:
        return _he2en[surface][0]
    before = [t for t in neighbor_tokens if t]
    # simple enrichment: take first 2 different neighbors
    enrich = ' '.join(before[:2]) if before else ''
    return ' '.join([surface, enrich]).strip()


def _cache_get(surface: str) -> np.ndarray | None:
    if surface in _embed_cache:
        _perf_stats['cache_hits'] += 1
        try:
            _embed_lru.remove(surface)
        except ValueError:
            pass
        _embed_lru.append(surface)
        return _embed_cache[surface]
    _perf_stats['cache_misses'] += 1
    return None

def _cache_put(surface: str, emb: np.ndarray):
    if surface in _embed_cache:
        try:
            _embed_lru.remove(surface)
        except ValueError:
            pass
    _embed_cache[surface] = emb
    _embed_lru.append(surface)
    while len(_embed_lru) > _EMBED_CACHE_MAX:
        old = _embed_lru.popleft()
        _embed_cache.pop(old, None)

def vector_candidates_faiss(line: Dict, topk: int) -> List[VectorHit]:
    base = Path(__file__).parents[3]
    idx, meta, model = _load_index(base)
    if idx is None or meta is None or model is None or not (meta.get('rows') or meta.get('entries')):
        return []
    n0 = line.get('n0', {})
    t1 = line.get('t1', {})
    tokens = t1.get('tokens', [])
    canonical_terms = (line.get('llm_aug') or {}).get('canonical_terms') or []

    hits: List[VectorHit] = []
    # Collect batch texts (VEC-1)
    batch_items: List[Tuple[Dict,str,str]] = []  # (tok, surface, qtext)
    for tok in tokens:
        if tok.get('kind') in {'number','pair','unit'}:
            continue
        surface = tok['text']
        idx_tok = tok['idx']
        neigh = [t['text'] for t in tokens if abs(t['idx']-idx_tok) <= 2 and t['idx'] != idx_tok]
        qtext = build_query_text(surface, neigh, canonical_terms)
        batch_items.append((tok, surface, qtext))
    # Build list of texts needing encode (skip cached)
    encode_texts = []
    encode_indices = []
    for i,(tok, surface, qtext) in enumerate(batch_items):
        emb = _cache_get(qtext)
        if emb is None:
            encode_indices.append(i)
            encode_texts.append(qtext)
    if encode_texts:
        t_enc0 = time.time()
        batch_embs = model.encode(encode_texts)
        _perf_stats['encode_times'].append(time.time()-t_enc0)
        _perf_stats['batch_sizes'].append(len(encode_texts))
        for j, emb in zip(encode_indices, batch_embs):
            _cache_put(batch_items[j][2], emb.astype(np.float32))
    # Perform searches
    for tok, surface, qtext in batch_items:
        qemb = _cache_get(qtext)
        if qemb is None:
            # should not happen, but guard
            qemb = model.encode([qtext])[0].astype(np.float32)
        if qemb.ndim == 1:
            qemb_arr = np.expand_dims(qemb,0)
        else:
            qemb_arr = qemb
        D, I = idx.search(qemb_arr.astype(np.float32), topk)
        # (VEC-4) cosine re-rank: reconstruct vectors & compute cosine with query embedding
        try:
            import numpy as _np
            qv = qemb_arr[0]
            qv_norm = qv / (_np.linalg.norm(qv) + 1e-9)
            cos_list = []
            for ii in I[0]:
                if ii < 0:
                    cos_list.append(None)
                    continue
                try:
                    vec = idx.reconstruct(int(ii))
                    vec_norm = vec / (_np.linalg.norm(vec) + 1e-9)
                    cos = float((qv_norm * vec_norm).sum())
                except Exception:
                    cos = None
                cos_list.append(cos)
        except Exception:
            cos_list = [None]*len(I[0])
        cand = []
        rows = meta.get('rows') or meta.get('entries') or []
        for score, ii, cos in zip(D[0], I[0], cos_list):
            if ii < 0 or ii >= len(rows):
                continue
            meta_row = rows[ii]
            final_vec = float(cos) if cos is not None else float(score)
            cand.append({'iri': meta_row['iri'], 'label': meta_row['label'], 'score_vec': final_vec, 'score_vec_raw': float(score)})
        # Re-rank by cosine if available
        if any(c.get('score_vec_raw') != c.get('score_vec') for c in cand):
            cand.sort(key=lambda x: x['score_vec'], reverse=True)
        hits.append(VectorHit(
            mention_id=f"v_{tok['idx']}",
            surface=surface,
            context=qtext,
            span=tuple(tok.get('span', (0,0))),
            candidates_vec=cand
        ))
    return hits


def run_vector_faiss(merged_block1_path: str, out_path: str) -> None:
    with open(merged_block1_path, 'r', encoding='utf-8') as fin, open(out_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            rec = json.loads(line)
            hits = vector_candidates_faiss(rec, CONFIG.TOPK_VEC)
            for h in hits:
                fout.write(h.model_dump_json()+'\n')
    # (VEC-6) write encode timing histogram sidecar
    try:
        times = _perf_stats.get('encode_times', [])
        if times:
            import math
            bins = [0]*5  # <10ms, <30ms, <60ms, <120ms, >=120ms
            for t in times:
                ms = t*1000.0
                if ms < 10: bins[0]+=1
                elif ms < 30: bins[1]+=1
                elif ms < 60: bins[2]+=1
                elif ms < 120: bins[3]+=1
                else: bins[4]+=1
            stats = {
                'encodes': len(times),
                'avg_ms': round(sum(times)/len(times)*1000.0,2),
                'p95_ms': round(sorted(times)[int(math.ceil(0.95*len(times))-1)]*1000.0,2),
                'hist_bins': {
                    '<10ms': bins[0], '<30ms': bins[1], '<60ms': bins[2], '<120ms': bins[3], '>=120ms': bins[4]
                },
                'cache_hits': _perf_stats.get('cache_hits'),
                'cache_misses': _perf_stats.get('cache_misses'),
                'batch_sizes': _perf_stats.get('batch_sizes')
            }
            Path(out_path).with_suffix('.vec_stats.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')
    except Exception:
        pass
