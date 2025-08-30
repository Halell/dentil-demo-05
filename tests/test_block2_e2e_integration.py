import json, time, tempfile
from pathlib import Path
from block2.src.pipeline.router_block2 import run_block2_all
from block2.src.pipeline import cand_faiss as cf
from scripts.prewarm_embeddings import main as prewarm

LEXICON = Path('artifacts/lexicon/ohd_lexicon.jsonl')


def _sample_real_label():
    if not LEXICON.exists():
        return 'implant'
    with open(LEXICON,'r',encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            iri = obj.get('iri','')
            label = obj.get('label','')
            if iri.startswith('http://purl.obolibrary.org/obo/') and label and ' ' not in label.lower():
                return label.lower()
    return 'implant'


def _tokens_for(text: str):
    toks = []
    pos = 0
    parts = text.split()
    for idx, p in enumerate(parts):
        span = (pos, pos+len(p))
        toks.append({"idx": idx, "text": p, "span": [span[0], span[1]], "kind": "word"})
        pos += len(p) + 1
    return toks


def test_h_integration_end_to_end():
    real_label = _sample_real_label()
    lines = [real_label, 'מולטיוניט', 'שלת', '12345']
    tmp_block1 = Path(tempfile.mkdtemp()) / 'block1_merged.jsonl'
    with open(tmp_block1,'w',encoding='utf-8') as f:
        for i, text in enumerate(lines):
            rec = {"id": i, "raw": text, "t1": {"tokens": _tokens_for(text)}}
            f.write(json.dumps(rec, ensure_ascii=False)+'\n')
    out_dir = Path(tempfile.mkdtemp())
    # Prewarm embeddings (loads model & first encode)
    prewarm()
    _ = run_block2_all(str(tmp_block1), str(out_dir / 'warm'))
    t0 = time.time()
    res = run_block2_all(str(tmp_block1), str(out_dir / 'timed'))
    dt = time.time() - t0
    per_line = dt / max(1,len(lines))
    # H3 performance: per line may still include FAISS search & encode loops; check avg encode time
    enc_times = cf._perf_stats['encode_times']
    if enc_times:
        avg_enc = sum(enc_times)/len(enc_times)
        assert avg_enc < 0.15, f"Avg encode took {avg_enc:.3f}s (>150ms)"  # GPU/CPU acceptable threshold
    merged = Path(res['merged'])
    assert merged.exists()
    # H1 verify at least one real OBO IRI present
    has_real = False
    with open(merged,'r',encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            for c in obj.get('candidates', []):
                if c.get('iri','').startswith('http://purl.obolibrary.org/obo/'):
                    has_real = True
                    break
    assert has_real, 'No real OBO IRIs in merged candidates'
    # H2 index health (ntotal>0)
    idx, meta, model = cf._load_index(Path(__file__).parents[1])
    if idx is not None:
        assert getattr(idx,'ntotal',0) > 0
