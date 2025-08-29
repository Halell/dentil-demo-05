from block1.src.pipeline.n0_normalize import normalize_n0

def test_n0_keeps_multiunit_word():
    text = "מולטיוניט שתל14 18/0"
    res = normalize_n0(text)
    assert res.normalized_text == "מולטיוניט שתל 14 18/0"
    assert 'split_glued_hebrew' not in res.notes
    # numbers extracted
    assert '14' in res.numbers and '18' in res.numbers and '0' in res.numbers
    # pair preserved
    assert any(p.A=='18' and p.B=='0' for p in res.pairs)
