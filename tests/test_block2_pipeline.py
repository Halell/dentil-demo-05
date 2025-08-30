import json, tempfile, os, shutil, pathlib
from pathlib import Path

# Basic tests for Block2 components focusing on tasks G1-G6
# Assumptions: artifacts/lexicon/ohd_lexicon.jsonl and FAISS index already built.

BASE = Path(__file__).parent.parent
BLOCK2 = BASE / 'block2' / 'src'
PIPE = BLOCK2 / 'pipeline'
COMMON = BLOCK2 / 'src' / 'common'

from block2.src.pipeline.t2_gazetteer import run_gazetteer
from block2.src.pipeline.cand_faiss import run_vector_faiss
from block2.src.pipeline.hybrid_ranker import merge_and_rank
from block2.src.pipeline.llm_rescue import llm_canonicalize_safe
from block2.src.common.config import CONFIG

LEX_PATH = BASE / 'artifacts' / 'lexicon' / 'ohd_lexicon.jsonl'


def _write_block1_lines(lines):
    tmp = Path(tempfile.mkstemp(suffix='.jsonl')[1])
    with open(tmp, 'w', encoding='utf-8') as f:
        for idx, text in enumerate(lines):
            rec = {"id": idx, "raw": text, "t1": {"tokens": [
                {"idx":0, "text": text, "span": [0, len(text)], "kind":"word"}
            ]}}
            f.write(json.dumps(rec, ensure_ascii=False)+'\n')
    return tmp


def test_g1_multiunit_alias_confident_singleton():
    line_file = _write_block1_lines(["מולטיוניט"])
    out_dir = Path(tempfile.mkdtemp())
    gaz = out_dir / 'gaz.jsonl'
    vec = out_dir / 'vec.jsonl'
    merged = out_dir / 'merged.jsonl'
    run_gazetteer(str(line_file), str(gaz))
    run_vector_faiss(str(line_file), str(vec))
    merge_and_rank(str(gaz), str(vec), str(merged))
    found = False
    with open(merged, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            if obj.get('surface') == 'מולטיוניט':
                cand = obj['candidates'][0]
                assert cand.get('score_lex',0) >= 0.9
                if obj.get('confident_singleton'):
                    found = True
    assert found


def test_g2_shalat_fuzzy_score_band():
    line_file = _write_block1_lines(["שלת"])
    out_dir = Path(tempfile.mkdtemp())
    gaz = out_dir / 'gaz.jsonl'
    vec = out_dir / 'vec.jsonl'
    merged = out_dir / 'merged.jsonl'
    run_gazetteer(str(line_file), str(gaz))
    run_vector_faiss(str(line_file), str(vec))
    merge_and_rank(str(gaz), str(vec), str(merged))
    with open(merged, 'r', encoding='utf-8') as f:
        data = [json.loads(l) for l in f]
    # find fuzzy candidate
    any_fuzzy = False
    for mc in data:
        for c in mc['candidates']:
            if c.get('score_lex') and 0.70 <= c['score_lex'] <= 0.85:
                any_fuzzy = True
    assert any_fuzzy


def test_g5_numbers_filtered():
    # number should not produce mention
    line_file = _write_block1_lines(["12345"])
    out_dir = Path(tempfile.mkdtemp())
    gaz = out_dir / 'gaz.jsonl'
    run_gazetteer(str(line_file), str(gaz))
    with open(gaz, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    assert len(lines) == 0


def test_llm_rescue_mock_returns_none_for_irrelevant():
    assert llm_canonicalize_safe('nonsense token that should not map') is None


def test_g4_no_vector_index_score_equals_lex(monkeypatch):
    # Monkeypatch vector loader to simulate missing index
    from block2.src.pipeline import cand_faiss as cf
    cf._index_cache = None; cf._meta_cache = None; cf._model_cache = None
    def fake_load(base):
        return None, None, None
    monkeypatch.setattr(cf, '_load_index', fake_load)
    line_file = _write_block1_lines(["מולטיוניט"])
    out_dir = Path(tempfile.mkdtemp())
    gaz = out_dir / 'gaz.jsonl'
    vec = out_dir / 'vec.jsonl'
    merged = out_dir / 'merged.jsonl'
    run_gazetteer(str(line_file), str(gaz))
    run_vector_faiss(str(line_file), str(vec))  # will produce empty
    merge_and_rank(str(gaz), str(vec), str(merged))
    with open(merged,'r',encoding='utf-8') as f:
        data=[json.loads(l) for l in f]
    for mc in data:
        for c in mc['candidates']:
            assert c['score_final'] == c.get('score_lex',0)


def test_g6_reweight_without_vec_same_as_lex():
    # Provide a token that matches lexicon only
    line_file = _write_block1_lines(["מולטיוניט"])
    out_dir = Path(tempfile.mkdtemp())
    gaz = out_dir / 'gaz.jsonl'
    vec = out_dir / 'vec.jsonl'
    merged = out_dir / 'merged.jsonl'
    run_gazetteer(str(line_file), str(gaz))
    # Skip vector run to simulate no vector evidence
    open(vec,'w').close()
    merge_and_rank(str(gaz), str(vec), str(merged))
    with open(merged,'r',encoding='utf-8') as f:
        data=[json.loads(l) for l in f]
    for mc in data:
        for c in mc['candidates']:
            assert c['score_final'] == c.get('score_lex',0)


def test_g3_vector_only_then_rescue(monkeypatch):
    # Force gazetteer to produce nothing by monkeypatching build_gazetteer to empty index
    import block2.src.pipeline.t2_gazetteer as tg
    class EmptyG:
        def extract(self, text):
            return []
        entries = {}
    def fake_build(*a, **k):
        return EmptyG()
    monkeypatch.setattr(tg, 'build_gazetteer', fake_build)
    # Force empty vector index first, then ensure rescue returns canonical terms (mocked)
    from block2.src.pipeline.router_block2 import run_block2_all
    import block2.src.pipeline.cand_faiss as cf
    def fake_load_index(base):
        return None, None, None
    monkeypatch.setattr(cf, '_load_index', fake_load_index)
    os.environ['CAND_ENGINE'] = 'faiss'
    line_file = _write_block1_lines(["שתל"])
    out_dir = Path(tempfile.mkdtemp())
    res = run_block2_all(str(line_file), str(out_dir))
    assert Path(res['merged']).exists()

