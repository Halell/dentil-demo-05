from __future__ import annotations
from pathlib import Path
from .t2_gazetteer import run_gazetteer
from .cand_faiss import run_vector_faiss
from .cand_neo4j import run_vector_neo4j
from ..common.config import CONFIG
from .hybrid_ranker import merge_and_rank
from .mv_candidates_builder import build_mv_v2

def run_block2_all(merged_block1_file: str, output_dir: str):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    gaz_file = out / 'gazetteer_hits.jsonl'
    vec_file = out / 'vector_hits.jsonl'
    merged_file = out / 'merged_candidates.jsonl'
    mv_file = out / 'mv_candidates_v2.jsonl'

    run_gazetteer(merged_block1_file, str(gaz_file))
    if CONFIG.CAND_ENGINE == 'neo4j':
        run_vector_neo4j(merged_block1_file, str(vec_file))
    else:
        run_vector_faiss(merged_block1_file, str(vec_file))
    merge_and_rank(str(gaz_file), str(vec_file), str(merged_file))
    build_mv_v2(str(merged_file), str(mv_file))
    return {
        'gazetteer': str(gaz_file),
        'vector': str(vec_file),
        'merged': str(merged_file),
        'mv': str(mv_file)
    }
