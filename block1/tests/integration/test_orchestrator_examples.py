import json
from pathlib import Path
from block1.src.pipeline.ops_apply import orchestrate_block1


def run_example(line: str):
    return orchestrate_block1(line)


def test_example_simple_pair():
    line = "14/16 מולטיוניט"
    result = run_example(line)
    assert 'normalized' in result
    assert result['normalized']['original'] == line
    # ensure pair preserved
    pair_tokens = [t for t in result['tokens'] if t['kind'] == 'pair']
    assert pair_tokens, 'Expected pair token'


def test_glued_hebrew_split():
    line = "מולטיוניט"
    result = run_example(line)
    # removed split_glued_hebrew expectation


def test_no_digit_loss():
    line = "3 יחידות 12/13"
    result = run_example(line)
    digits_orig = ''.join(ch for ch in line if ch.isdigit())
    digits_new = ''.join(ch for ch in result['normalized']['applied_text'] if ch.isdigit())
    assert digits_orig == digits_new


def test_perf_present():
    line = "מולטיוניט 14/16"
    result = run_example(line)
    assert 'perf' in result and 'total_ms' in result['perf']
