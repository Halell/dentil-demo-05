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
        return [0.0 for _ in vals]  # avoid inflating single candidate
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

    # Score normalization and final scoring
    for mc in merged.values():
        lex_scores = [c.get('score_lex', 0.0) for c in mc.candidates]
        vec_scores = [c.get('score_vec', 0.0) for c in mc.candidates]
        norm_lex = _norm_scores(lex_scores)
        norm_vec = _norm_scores(vec_scores)
        for idx, c in enumerate(mc.candidates):
            c['norm_lex'] = norm_lex[idx] if idx < len(norm_lex) else 0.0
            c['norm_vec'] = norm_vec[idx] if idx < len(norm_vec) else 0.0
            c['score_final'] = (
                cfg.W_LEX * c.get('norm_lex', 0.0) +
                cfg.W_VEC * c.get('norm_vec', 0.0)
            )
        # sort candidates
        mc.candidates.sort(key=lambda x: x['score_final'], reverse=True)
        # determine singleton confidence
        if mc.candidates and mc.candidates[0]['score_final'] >= cfg.TAU_CONFIDENT and (len(mc.candidates) == 1 or (len(mc.candidates)>1 and mc.candidates[0]['score_final'] - mc.candidates[1]['score_final'] > 0.2)):
            mc.confident_singleton = True
        # truncate
        mc.candidates = mc.candidates[:cfg.TOPK_FINAL]

    with open(out_path, 'w', encoding='utf-8') as f:
        for mc in merged.values():
            f.write(mc.model_dump_json()+'\n')
