from block1.src.pipeline.n0_normalize import normalize_n0
from block1.src.pipeline.t1_tokenize import tokenize_t1

def test_t1_tokenizes_multiunit_whole():
    text = "מולטיוניט שתל14 18/0"
    n0 = normalize_n0(text)
    t1 = tokenize_t1(n0)
    words = [t.text for t in t1.tokens]
    assert "מולטיוניט" in words
    # ensure not split into parts like 'מולטי' 'וניט'
    assert not ('מולטי' in words and 'וניט' in words)
    # pair token present
    assert any(t.kind=='pair' and t.text.startswith('18/0') for t in t1.tokens)
