"""Bundling utilities (BDL-1..3).
BDL-1: build bundle (device + dimension tokens)
BDL-2: merge sequential mentions with same IRI (span expansion)
BDL-3: prune vector-only low score overlapping lex
"""
from __future__ import annotations
from typing import List, Dict

def merge_sequential_same_iri(mentions: List[Dict]) -> List[Dict]:
    mentions_sorted = sorted(mentions, key=lambda m: m['span'][0])
    out = []
    for m in mentions_sorted:
        if not out:
            out.append(m); continue
        last = out[-1]
        if last['candidates'][0]['iri'] == m['candidates'][0]['iri'] and last['span'][1] == m['span'][0]:
            # extend
            last['span'] = (last['span'][0], m['span'][1])
            last['surface'] = last['surface'] + ' ' + m['surface']
        else:
            out.append(m)
    return out

def build_device_dimension_bundles(mentions: List[Dict]) -> List[Dict]:
    # naive: if a DEVICE/IMPLANT mention followed by number/pair tokens inside surface create bundle label
    bundles = []
    for m in mentions:
        iri = m['candidates'][0]['iri'] if m.get('candidates') else ''
        if iri.startswith(('DEVICE:','IMPLANT:')) and any(ch.isdigit() for ch in m['surface']):
            b = dict(m)
            b['bundle'] = True
            bundles.append(b)
    return bundles

def prune_vector_only(mentions: List[Dict], threshold: float=0.15) -> List[Dict]:
    out = []
    for m in mentions:
        if not m.get('candidates'):
            continue
        cand0 = m['candidates'][0]
        if cand0.get('score_lex',0.0)==0 and cand0.get('score_final',0.0) < threshold:
            continue
        out.append(m)
    return out

def bundle_pipeline(mentions: List[Dict]) -> List[Dict]:
    merged = merge_sequential_same_iri(mentions)
    merged = prune_vector_only(merged)
    bundles = build_device_dimension_bundles(merged)
    return merged + bundles