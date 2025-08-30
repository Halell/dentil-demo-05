from __future__ import annotations
import json, os
from typing import List, Dict
from ..common.config import CONFIG
from ..common.schemas import VectorHit

try:
    from neo4j import GraphDatabase  # type: ignore
except Exception:
    GraphDatabase = None  # type: ignore

_driver = None
_meta_cache = None

def _get_driver():
    global _driver
    if _driver is None and GraphDatabase is not None:
        try:
            _driver = GraphDatabase.driver(
                CONFIG.NEO4J_URI,
                auth=(CONFIG.NEO4J_USER, CONFIG.NEO4J_PASS)
            )
        except Exception:
            _driver = None
    return _driver

def _load_meta(tx):
    # Basic metadata: iri,label,embedding length
    q = """
    MATCH (c:Concept) WHERE exists(c.embedding) RETURN c.iri as iri, c.label as label LIMIT 1
    """
    rec = tx.run(q).single()
    if rec:
        return True
    return False

def _vector_search(tx, query_vec: List[float], topk: int):
    # Requires vector index named CONFIG.NEO4J_VECTOR_INDEX
    q = f"""
    CALL db.index.vector.queryNodes('{CONFIG.NEO4J_VECTOR_INDEX}', $topk, $qv) YIELD node, score
    RETURN node.iri AS iri, node.label AS label, score AS score
    """
    try:
        return [r.data() for r in tx.run(q, topk=topk, qv=query_vec)]
    except Exception:
        return []

def vector_candidates_neo4j(line: Dict, topk: int) -> List[VectorHit]:
    driver = _get_driver()
    if driver is None:
        return []
    tokens = (line.get('t1') or {}).get('tokens', [])
    canonical_terms = (line.get('llm_aug') or {}).get('canonical_terms') or []
    hits: List[VectorHit] = []
    # Simple lazy embed model reuse from FAISS module if available
    try:
        from .cand_faiss import _model_cache, build_query_text
    except Exception:
        build_query_text = lambda s, n, c: s  # type: ignore
        _model_cache = None  # type: ignore
    if _model_cache is None:
        return []
    with driver.session() as sess:
        for tok in tokens:
            if tok.get('kind') in {'number','pair','unit'}:
                continue
            surface = tok['text']
            idx_tok = tok['idx']
            neigh = [t['text'] for t in tokens if abs(t['idx']-idx_tok) <= 2 and t['idx'] != idx_tok]
            qtext = build_query_text(surface, neigh, canonical_terms)
            try:
                emb = _model_cache.encode([qtext])[0]
                qv = emb.tolist()
            except Exception:
                continue
            rows = sess.execute_read(lambda tx: _vector_search(tx, qv, topk))
            cand = []
            for r in rows:
                cand.append({
                    'iri': r.get('iri'),
                    'label': r.get('label'),
                    'score_vec': float(r.get('score', 0.0))
                })
            hits.append(VectorHit(
                mention_id=f"v_{tok['idx']}",
                surface=surface,
                context=qtext,
                span=tuple(tok.get('span',(0,0))),
                candidates_vec=cand
            ))
    return hits

def run_vector_neo4j(merged_block1_path: str, out_path: str) -> None:
    driver = _get_driver()
    with open(merged_block1_path, 'r', encoding='utf-8') as fin, open(out_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            rec = json.loads(line)
            hits = vector_candidates_neo4j(rec, CONFIG.TOPK_VEC)
            for h in hits:
                fout.write(h.model_dump_json()+'\n')
