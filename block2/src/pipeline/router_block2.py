from __future__ import annotations
from pathlib import Path
from .t2_gazetteer import run_gazetteer
from .cand_faiss import run_vector_faiss
from .cand_neo4j import run_vector_neo4j
from ..common.config import CONFIG
from .hybrid_ranker import merge_and_rank
from .mv_candidates_builder import build_mv_v2
from .llm_rescue import llm_canonicalize_safe
from .bundler import bundle_pipeline
import json

def _needs_rescue(g_path: str, v_path: str) -> bool:
    # Rescue if both gazetteer and vector output are empty
    try:
        if (Path(g_path).stat().st_size == 0) and (Path(v_path).stat().st_size == 0):
            return True
    except Exception:
        return True
    return False

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
    # LLM Rescue pass (only if both empty and feature desired)
    if CONFIG.FEATURES.get('llm_rescue', True) and _needs_rescue(gaz_file, vec_file):
        # create augmented temp file with canonical terms then re-run vector
        tmp_aug = out / 'block1_augmented.jsonl'
        with open(merged_block1_file, 'r', encoding='utf-8') as fin, open(tmp_aug, 'w', encoding='utf-8') as fout:
            for line in fin:
                rec = json.loads(line)
                text = rec.get('raw') or rec.get('text') or ''
                aug = llm_canonicalize_safe(text)
                if aug:
                    rec['llm_aug'] = aug
                fout.write(json.dumps(rec, ensure_ascii=False)+'\n')
        # Re-run vector over augmented content (append results)
        if CONFIG.CAND_ENGINE == 'neo4j':
            run_vector_neo4j(str(tmp_aug), str(vec_file))
        else:
            run_vector_faiss(str(tmp_aug), str(vec_file))
    merge_and_rank(str(gaz_file), str(vec_file), str(merged_file))
    build_mv_v2(str(merged_file), str(mv_file))
    # Optional bundling artifact
    bundle_file = out / 'bundled_candidates.jsonl'
    try:
        with open(mv_file, 'r', encoding='utf-8') as fin, open(bundle_file,'w',encoding='utf-8') as fout:
            for line in fin:
                obj = json.loads(line)
                bundled = bundle_pipeline([obj])  # per line mention list minimal; pipeline expects list
                for b in bundled:
                    fout.write(json.dumps(b, ensure_ascii=False)+'\n')
    except Exception:
        bundle_file = None
    return {
        'gazetteer': str(gaz_file),
        'vector': str(vec_file),
        'merged': str(merged_file),
        'mv': str(mv_file),
        'bundled': str(bundle_file) if bundle_file else None
    }
