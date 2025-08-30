"""HY-4: Simple calibration script to tune W_LEX/W_VEC/W_PRIOR/W_CTX via grid search.
Usage: python -m block2.scripts.calibrate_weights --gold gold.jsonl --candidates merged_candidates.jsonl
Gold format: each line JSON {"surface":..., "iri":...}
"""
from __future__ import annotations
import json, argparse, itertools, statistics
from pathlib import Path

def load_gold(path: str):
    gold = {}
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            gold[obj.get('surface','').lower()] = obj.get('iri')
    return gold

def load_candidates(path: str):
    out = []
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out

def evaluate(cands, gold, weights):
    W_LEX, W_VEC, W_PRIOR, W_CTX = weights
    correct = 0; total=0
    for mc in cands:
        surface = mc.get('surface','').lower()
        expected = gold.get(surface)
        if not expected:
            continue
        best = None; best_score=-1
        for c in mc.get('candidates', []):
            lex = c.get('score_lex',0.0)
            vec = c.get('norm_vec',0.0)
            prior = c.get('score_prior',0.0)
            # heuristic: treat context boost already inside final; ignore here
            score = W_LEX*lex + W_VEC*vec + W_PRIOR*prior
            if score > best_score:
                best_score = score; best = c
        if best and best.get('iri') == expected:
            correct +=1
        total +=1
    return correct/total if total else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gold', required=True)
    ap.add_argument('--candidates', required=True)
    args = ap.parse_args()
    gold = load_gold(args.gold)
    cands = load_candidates(args.candidates)
    grid_lex = [0.4,0.6,0.8]
    grid_vec = [0.1,0.2,0.3]
    grid_prior = [0.0,0.05,0.1]
    grid_ctx = [0.0,0.04]
    best = None; best_score=-1
    for comb in itertools.product(grid_lex, grid_vec, grid_prior, grid_ctx):
        score = evaluate(cands, gold, comb)
        if score > best_score:
            best_score = score; best = comb
    print(json.dumps({
        'best_weights': {
            'W_LEX': best[0], 'W_VEC': best[1], 'W_PRIOR': best[2], 'W_CTX': best[3]
        },
        'accuracy': best_score
    }, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()