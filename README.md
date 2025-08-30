# OHD Clinic NLP (Block 0)

Setup skeleton for ontology lexicon + embeddings + Ollama wrapper.

## Quick Start

1. python -m venv .venv
2. Activate venv
3. pip install -r requirements.txt
4. Copy .env.example to .env and adjust paths.
5. Run: python src/ontology/build_lexicon.py
6. Run: python src/ontology/build_embeddings.py
7. Sanity: python src/tools/sanity_checks.py

Artifacts under artifacts/ .

## Block 2 (Entity Candidates) – Scoring & Rescue

Pipeline stages:
1. Gazetteer (fast path single-token, Hebrew normalization + fuzzy Damerau ≤1 edit).
2. Vector retrieval (FAISS; query enriched with neighbors / optional LLM canonical terms).
3. Hybrid merge: lexical scores kept raw (label=1.0, synonym=0.9, fuzzy mapped 0.70–0.85). Vector scores normalized per mention then weighted (default W_LEX 0.6 / W_VEC 0.3).
4. Confident singleton: one strong lexical (≥0.9) prunes vector-only distractors.
5. LLM Rescue (optional): if both gazetteer & vector empty, a guarded canonicalizer proposes up to 3 safe English terms; vector re-run uses them.

Other rules:
- Placeholder IRIs (IMPLANT:, DEVICE:, HE2EN:) dropped if a real OBO IRI present for same span.
- Numeric / unit / pair tokens filtered out in both layers.
- covered_token_idxs tracks token coverage for future pruning.

Env toggles:
- FEAT_LLM_RESCUE=0 disables rescue.

Outputs written under block2/output/: gazetteer_hits.jsonl, vector_hits.jsonl, merged_candidates.jsonl, mv_candidates_v2.jsonl.

## Turbo Chat Example

```python
from src.llm.ollama_client import chat_json
resp = chat_json('{"ping":1}', system='return JSON', turbo=True)
print(resp)
```
