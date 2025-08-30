from __future__ import annotations
"""Diagnostic script:
1. Gazetteer coverage vs OHD lexicon (labels only, excluding numeric/short).
2. Sample vector retrieval for a few labels using FAISS.
3. LLM rescue augmentation demo.

Run: python scripts/diagnose_gazetteer_and_vector.py
Outputs human-readable summary.
"""
import json, random, sys, os
from pathlib import Path
from typing import List, Dict

# Ensure repository root on path so that 'block2.src' imports resolve
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from block2.src.pipeline.t2_gazetteer import build_gazetteer, normalize_hebrew  # type: ignore
from block2.src.pipeline.cand_faiss import vector_candidates_faiss  # type: ignore
from block2.src.pipeline.llm_rescue import llm_canonicalize_safe  # type: ignore

LEX_PATH = Path('artifacts/lexicon/ohd_lexicon.jsonl')
CLINIC = 'block2/data/dictionaries/clinic_abbreviations.json'
BRANDS = 'block2/data/dictionaries/brand_names.json'

def load_lexicon_records() -> List[Dict]:
    recs = []
    if not LEX_PATH.exists():
        print(f"ERROR: lexicon not found at {LEX_PATH}")
        return recs
    with open(LEX_PATH,'r',encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get('iri') and rec.get('label'):
                recs.append(rec)
    return recs

def step1_gazetteer_coverage():
    lex = load_lexicon_records()
    g = build_gazetteer(str(LEX_PATH), CLINIC, BRANDS)
    entries = g.entries
    total_considered = 0
    missing = []
    for rec in lex:
        label = rec['label']
        if label.isdigit() or len(label) <= 2:
            continue  # excluded by design
        total_considered += 1
        surf_key = normalize_hebrew(label.lower())
        if surf_key not in entries:
            missing.append({'iri': rec['iri'], 'label': label})
    return {
        'total_labels_in_lexicon': len(lex),
        'considered_labels': total_considered,
        'gazetteer_surfaces': len(entries),
        'missing_count': len(missing),
        'missing_sample': missing[:5]
    }

def _make_synthetic_line(text: str) -> Dict:
    # minimal line structure for vector_candidates_faiss
    tokens = [{'idx':0,'text':text,'kind':'word','span':[0,len(text)]}]
    return {'t1': {'tokens': tokens}, 'n0': {'normalized_text': text}}

def step2_vector_samples(sample_size: int = 5):
    lex = load_lexicon_records()
    if not lex:
        return {'error': 'no lexicon'}
    # choose random subset of medium-length labels
    candidates = [r for r in lex if 4 <= len(r['label']) <= 25]
    random.shuffle(candidates)
    chosen = candidates[:sample_size]
    out = []
    for rec in chosen:
        line = _make_synthetic_line(rec['label'])
        vec_hits = vector_candidates_faiss(line, topk=3)
        if vec_hits:
            first = vec_hits[0]
            cands = []
            for vh in vec_hits:
                cands.extend(vh.candidates_vec)
            top3 = cands[:3]
            out.append({
                'query': rec['label'],
                'top_iris': [(c['iri'], round(c.get('score_vec',0),4)) for c in top3]
            })
        else:
            out.append({'query': rec['label'], 'top_iris': []})
    return out

def step3_llm_demo():
    examples = ['מולטיוניט', 'שתל', 'בלארגז']
    results = []
    for ex in examples:
        aug = llm_canonicalize_safe(ex)
        results.append({'input': ex, 'llm_aug': aug})
    return results

def main():
    report = {
        'gazetteer_coverage': step1_gazetteer_coverage(),
        'vector_samples': step2_vector_samples(),
        'llm_demo': step3_llm_demo()
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
