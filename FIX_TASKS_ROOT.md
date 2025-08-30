# Task Pack – Consolidated Follow‑Up Items (Blocks 1 & 2)

## Legend
- [ ] Open  |  [x] Done  |  ~ Partial  | (P) Planned / not started

## 1. Gazetteer & Alias Layer
- [x] GZ-1 Allow multi-word n-gram before single-token cover (optional mode)
- [x] GZ-2 Keep multiple metas per surface (config toggle `KEEP_ALL_METAS`)
- [x] GZ-3 Post-merge span grouping (attach adjacent Hebrew device + size tokens)
- [x] GZ-4 Add frequency prior (count occurrences across corpus) -> `score_prior`
- [x] GZ-5 Export coverage stats (tokens covered %, avg candidates per mention)

## 2. Fuzzy & Normalization
- [x] FZ-1 Reinstate direct variant for "שלת" but still record fuzzy_score source
- [x] FZ-2 Add phonetic code (e.g., double-metaphone for transliterated EN) to boost near-miss
- [x] FZ-3 Adaptive fuzzy band: widen to (0.65–0.88) if edit distance=1 and length<=4

## 3. Vector / Embeddings
 - [x] VEC-1 Batch encode all tokens per line (reduce per-line latency)
 - [x] VEC-2 Add surface embedding cache across lines (LRU size configurable)
 - [x] VEC-3 Offline model packaging (copy model into `artifacts/models/` + path swap)
 - [x] VEC-4 Add cosine similarity re-rank within top FAISS hits
 - [x] VEC-5 Implement Neo4j vector backend parity (cand_neo4j)
 - [x] VEC-6 Add encode timing histogram to logs

## 4. Hybrid Scoring
- [x] HY-1 Introduce prior weight W_PRIOR using `score_prior`
- [x] HY-2 Context window semantic boost: if neighbor token has material hint & candidate material
- [x] HY-3 Penalize placeholder IRIs globally (e.g., -0.05) rather than only dedup removal
- [x] HY-4 Add calibration script to tune W_LEX/W_VEC via small labeled set

## 5. LLM Rescue
- [x] LLM-1 Real LLM integration (external service) with strict JSON schema validation (placeholder hook)
- [x] LLM-2 Add guardrail: max 2 network retries, latency metric
- [x] LLM-3 Cache canonicalization results by normalized surface

## 6. Bundling / Mention Aggregation
- [x] BDL-1 Implement bundle builder (device + dimension pair)
- [x] BDL-2 Merge sequential mentions with same IRI (span expansion)
- [x] BDL-3 Pruner: remove vector-only mentions with score_final < 0.15 if overlapping lex

## 7. Quality & Testing
- [ ] QA-1 Add golden test fixtures (small labeled corpus) for precision/recall report
- [ ] QA-2 Property-based fuzz test for Hebrew normalization invariants
- [ ] QA-3 Performance test: ensure avg encode < 80ms on GPU / < 180ms CPU
- [ ] QA-4 Stress test with 5K lines (memory & cumulative latency stats)
- [ ] QA-5 Static type pass (enable mypy / pydantic strict mode)

## 8. Tooling & DX
- [ ] DX-1 Add CLI command `block2 stats` to print coverage & timing
- [ ] DX-2 Add environment flag to dump intermediate enriched query text for debug
- [ ] DX-3 Improve logging format (structured JSON lines)

## 9. Documentation
- [ ] DOC-1 Expand README: architecture diagram (Block1→Block2)
- [ ] DOC-2 Create FAQ (fuzzy vs exact, why placeholder IRIs appear)
- [ ] DOC-3 Add CONTRIBUTING.md with coding standards

## 10. Deployment
- [ ] DEP-1 Dockerfile slim (multi-stage, CPU + optional CUDA variant)
- [ ] DEP-2 Preload embeddings & run prewarm in container entrypoint
- [ ] DEP-3 Health endpoint (FastAPI) returning index stats & version

---
Generated baseline backlog from current code gaps and earlier discussion. Prioritize by marking top 5 and we can break them into implementation PRs.
