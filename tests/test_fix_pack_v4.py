import json, tempfile
from pathlib import Path

from block2.src.pipeline.router_block2 import run_block2_all
from block2.src.pipeline.hybrid_ranker import merge_and_rank
from block2.src.common.schemas import GazetteerHit, VectorHit


def _make_block1_file(lines):
    tmp = Path(tempfile.mkdtemp()) / 'block1_merged.jsonl'
    with open(tmp, 'w', encoding='utf-8') as f:
        for i, text in enumerate(lines):
            toks = [{"idx":0,"text":text,"span":[0,len(text)],"kind":"word"}]
            rec = {"id": i, "raw": text, "t1": {"tokens": toks}}
            f.write(json.dumps(rec, ensure_ascii=False)+'\n')
    return tmp


def _load_merged(path: Path):
    out = []
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            out.append(json.loads(line))
    return out


def test_v4_alias_resolution_and_no_placeholders():
    blk1 = _make_block1_file(["מולטיוניט", "שלת"])
    out_dir = Path(tempfile.mkdtemp())
    res = run_block2_all(str(blk1), str(out_dir))
    merged = Path(res['merged'])
    objs = _load_merged(merged)
    surfaces = {o['surface']: o for o in objs}
    for surface in ["מולטיוניט", "שלת"]:
        assert surface in surfaces, f"missing mention for {surface}"
        cand_iri = surfaces[surface]['candidates'][0]['iri']
        assert cand_iri.startswith('http://purl.obolibrary.org/obo/OHD_'), f"Expected OHD IRI, got {cand_iri}"
        assert 'DEVICE:' not in cand_iri and 'IMPLANT:' not in cand_iri


def test_v4_vector_support_multiunit_norm_vec():
    blk1 = _make_block1_file(["מולטיוניט"])
    out_dir = Path(tempfile.mkdtemp())
    res = run_block2_all(str(blk1), str(out_dir))
    merged = Path(res['merged'])
    objs = _load_merged(merged)
    for o in objs:
        if o['surface'] == 'מולטיוניט':
            # vector component should be present with norm_vec > 0 on at least one candidate
            assert any(c.get('norm_vec',0) > 0 for c in o['candidates']), 'Expected norm_vec > 0 for multiunit'


def test_v4_confident_singleton_source_allowed():
    blk1 = _make_block1_file(["מולטיוניט"])
    out_dir = Path(tempfile.mkdtemp())
    res = run_block2_all(str(blk1), str(out_dir))
    objs = _load_merged(Path(res['merged']))
    for o in objs:
        if o.get('confident_singleton'):
            src = o['candidates'][0].get('iri_source') or o['candidates'][0].get('source')
            assert src in {"ohd_label","ohd_synonym","resolved_alias"}, f"confident_singleton with disallowed source {src}"


def test_v4_hints_and_surface_preserved():
    blk1 = _make_block1_file(["מולטיוניט", "שלת"])
    out_dir = Path(tempfile.mkdtemp())
    res = run_block2_all(str(blk1), str(out_dir))
    objs = _load_merged(Path(res['merged']))
    for o in objs:
        if o['surface'] == 'מולטיוניט':
            assert 'device_hint' in o.get('hints', []), 'device_hint missing'
        if o['surface'] == 'שלת':
            assert 'implant_hint' in o.get('hints', []), 'implant_hint missing'
        # surface should equal original token text (no normalization override)
        assert o['surface'] in {'מולטיוניט','שלת'}


def test_v4_vector_query_uses_he2en_mapping():
    # Inspect vector hits context to ensure English mapping used
    blk1 = _make_block1_file(["מולטיוניט"])
    out_dir = Path(tempfile.mkdtemp())
    res = run_block2_all(str(blk1), str(out_dir))
    vec_file = Path(res['vector'])
    used_contexts = []
    with open(vec_file,'r',encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            if obj.get('surface') == 'מולטיוניט':
                used_contexts.append(obj.get('context',''))
    assert any('multi unit abutment' in c or 'multi-unit abutment' in c for c in used_contexts), 'Expected he2en English mapping in query text'


def test_v4_placeholder_penalty():
    # Construct artificial gazetteer hit with both real OHD and alias_only placeholder and ensure placeholder scores lower
    import uuid
    tmp_dir = Path(tempfile.mkdtemp())
    gaz_file = tmp_dir / 'gaz.jsonl'
    vec_file = tmp_dir / 'vec.jsonl'
    merged = tmp_dir / 'merged.jsonl'
    # Real candidate
    gh_real = GazetteerHit(
        mention_id='m0_0_5', surface='מולטיוניט', span=(0,5), ngram=1,
        candidates_lex=[{'iri':'http://purl.obolibrary.org/obo/OHD_0000001','label':'multi unit abutment','score_lex':1.0,'iri_source':'ohd_label'}],
        hints=['device_hint'], covered_token_idxs=[0]
    )
    # Placeholder alias_only
    gh_placeholder = GazetteerHit(
        mention_id='m0_6_11', surface='מולטיוניט', span=(6,11), ngram=1,
        candidates_lex=[{'iri':'DEVICE:multiunit_abutment','label':'multi unit abutment','score_lex':1.0,'iri_source':'alias_only'}],
        hints=['device_hint'], covered_token_idxs=[0]
    )
    with open(gaz_file,'w',encoding='utf-8') as f:
        f.write(gh_real.model_dump_json()+'\n')
        f.write(gh_placeholder.model_dump_json()+'\n')
    open(vec_file,'w').close()  # no vector evidence
    merge_and_rank(str(gaz_file), str(vec_file), str(merged))
    objs = _load_merged(merged)
    # flatten candidates
    cand_scores = {}
    for o in objs:
        for c in o['candidates']:
            cand_scores[c['iri']] = c['score_final']
    assert cand_scores['http://purl.obolibrary.org/obo/OHD_0000001'] >= cand_scores['DEVICE:multiunit_abutment'] + 0.05, 'Expected placeholder penalty'
