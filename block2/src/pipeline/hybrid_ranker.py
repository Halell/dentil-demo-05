from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Dict, List
from ..common.config import CONFIG
from ..common.schemas import GazetteerHit, VectorHit, MentionCandidates


def _load_gazetteer_hits(path: str) -> Dict[str, GazetteerHit]:
    hits = {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                obj = GazetteerHit.model_validate_json(line)
                hits[obj.mention_id] = obj
    except FileNotFoundError:
        pass
    return hits


def _load_vector_hits(path: str) -> List[VectorHit]:
    out = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                out.append(VectorHit.model_validate_json(line))
    except FileNotFoundError:
        pass
    return out


def _norm_scores(vals: List[float]) -> List[float]:
    if not vals:
        return []
    mx = max(vals)
    mn = min(vals)
    if mx == mn:
        return [0.0 for _ in vals]
    return [(v - mn)/(mx - mn) for v in vals]


def merge_and_rank(gazetteer_file: str, vector_file: str, out_path: str) -> None:
    cfg = CONFIG
    gaz_hits = _load_gazetteer_hits(gazetteer_file)
    vec_hits = _load_vector_hits(vector_file)

    # Build mention candidates seeded from lexical hits first
    merged: Dict[str, MentionCandidates] = {}
    for mid, gh in gaz_hits.items():
        candidates = []
        for c in gh.candidates_lex:
            candidates.append({
                'iri': c['iri'],
                'label': c['label'],
                'score_lex': c.get('score_lex', 0.0)
            })
        merged[mid] = MentionCandidates(
            mention_id=mid,
            surface=gh.surface,
            span=gh.span,
            hints=gh.hints,
            candidates=candidates,
        )

    # Attach vector suggestions (as separate mentions when not overlapping existing spans)
    existing_spans = [mc.span for mc in merged.values()]
    def overlaps(span, others):
        return any(not (span[1] <= o[0] or span[0] >= o[1]) for o in others)

    for vh in vec_hits:
        # Attempt to attach to an existing mention with same surface substring containment
        attached = False
        for mc in merged.values():
            if vh.surface.lower() in mc.surface.lower():
                # merge vector candidates into same mention candidate list
                for c in vh.candidates_vec:
                    mc.candidates.append({
                        'iri': c['iri'],
                        'label': c['label'],
                        'score_vec': c.get('score_vec', 0.0)
                    })
                attached = True
                break
        if attached:
            continue
        # create new mention if span not overlapping existing
        span = vh.span or (0,0)
        if span == (0,0) or not overlaps(span, existing_spans):
            candidates = []
            for c in vh.candidates_vec:
                candidates.append({'iri': c['iri'], 'label': c['label'], 'score_vec': c.get('score_vec', 0.0)})
            mid = vh.mention_id
            merged[mid] = MentionCandidates(
                mention_id=mid,
                surface=vh.surface,
                span=span,
                hints=[],
                candidates=candidates
            )

    # Score aggregation (lex direct, vector normalized, prior (HY-1), context boost (HY-2), placeholder penalty (HY-3))
    for mc in merged.values():
        vec_scores = [c.get('score_vec', 0.0) for c in mc.candidates if c.get('score_vec') is not None]
        norm_vec_all = _norm_scores(vec_scores) if any(vs > 0 for vs in vec_scores) else []
        # assign normalized vector scores preserving order
        vec_iter = iter(norm_vec_all)
        for c in mc.candidates:
            if c.get('score_vec') is not None and c.get('score_vec',0.0)>0:
                c['norm_vec'] = next(vec_iter, 0.0)
            else:
                c['norm_vec'] = 0.0
        for c in mc.candidates:
            lex = c.get('score_lex', 0.0)  # already 0..1
            vec = c.get('norm_vec', None)
            prior = c.get('score_prior', 0.0)
            # context boost placeholder (maintain existing behavior)
            ctx_component = 0.0
            # assemble parts present
            parts = {
                'lex': lex,
                'vec': vec if vec and vec>0 else None,
                'prior': prior if prior>0 else None,
            }
            present = {k:v for k,v in parts.items() if v is not None}
            weights = {
                'lex': cfg.W_LEX,
                'vec': cfg.W_VEC,
                'prior': cfg.W_PRIOR,
                'ctx': cfg.W_CTX,
            }
            Z = sum(weights[k] for k in present)
            raw = sum(weights[k]*present[k] for k in present) / (Z or 1e-9)
            # alias_only penalty
            iri_src = c.get('iri_source') or c.get('source')
            if iri_src == 'alias_only':
                raw = max(0.0, raw - 0.08)
            c['score_final'] = raw + ctx_component
        mc.candidates.sort(key=lambda x: x['score_final'], reverse=True)
        # confident singleton rules
        if len(mc.candidates) == 1:
            c0 = mc.candidates[0]
            iri_src = c0.get('iri_source') or c0.get('source')
            if c0.get('score_lex',0)>=0.9:
                mc.confident_singleton = True
        else:
            strong = [c for c in mc.candidates if c.get('score_lex',0)>=0.9]
            if len(strong) == 1:
                # keep only the strong lexical candidate
                mc.candidates = strong
                mc.confident_singleton = True
        # restrict topK
        mc.candidates = mc.candidates[:cfg.TOPK_FINAL]

    with open(out_path, 'w', encoding='utf-8') as f:
        for mc in merged.values():
            f.write(mc.model_dump_json()+'\n')
