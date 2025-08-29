"""Run the FULL Block1 pipeline (N0 -> T1 -> LLM refine -> LLM hints -> ops apply -> perf) on a single line.

Usage examples (from repo root with venv active):
  python -m block1.scripts.quick_full_pipeline_example
  python -m block1.scripts.quick_full_pipeline_example --line "מולטיוניט שתל14 18/0"
  python -m block1.scripts.quick_full_pipeline_example --line "טיפול 14-16" --no-llm

Flags:
  --line TEXT     Override default input line.
  --no-llm        Disable LLM stages (deterministic only).
  --print-mv      Print constructed Marked View (debug).
  --reasoning LVL Force reasoning level (high|medium|low).

Env toggle (alternative): set BLOCK1_NO_LLM=1 to disable LLM globally for this script.
"""
from __future__ import annotations
import os, json, argparse, sys
from pprint import pprint

from block1.src.pipeline.ops_apply import orchestrate_block1
from block1.src.pipeline.mv_builder import MVBuilder
from block1.src.pipeline.n0_normalize import normalize_n0
from block1.src.pipeline.t1_tokenize import tokenize_t1
from block1.src.pipeline.n0b_llm_refine import refine_with_llm
from block1.src.pipeline.t1b_llm_hints import generate_hints_with_llm

DEFAULT_LINE = "מולטיוניט שתל14 18/0"


def run_full(line: str, use_llm: bool, print_mv: bool, reasoning_level: str | None):
    # First deterministic stages for MV (so refine/hints get richer context if desired)
    n0 = normalize_n0(line)
    t1 = tokenize_t1(n0)
    mv = MVBuilder().build_marked_view(n0, t1, include_example=True, reasoning_level=reasoning_level)

    if print_mv:
        print("\n===== MARKED VIEW =====\n" + mv + "\n=======================\n")

    if not use_llm:
        # Deterministic only
        result = orchestrate_block1(line)
        result['mv'] = mv
        return result

    # Wrap LLM functions to supply MV context while keeping orchestrator contract
    def refine_adapter(text: str, tokens_result):
        # reuse MV we built (ignore text param) for best context
        out = refine_with_llm(mv, tokens_result)
        return out

    def hints_adapter(text: str, tokens_result):
        return generate_hints_with_llm(mv, tokens_result)

    result = orchestrate_block1(
        line,
        llm_refine_fn=refine_adapter,
        llm_hints_fn=hints_adapter,
        reasoning_level=reasoning_level,
    )
    result['mv'] = mv
    return result


def main():
    parser = argparse.ArgumentParser(description="Run full Block1 pipeline on one line")
    parser.add_argument('--line', type=str, default=DEFAULT_LINE, help='Input line')
    parser.add_argument('--no-llm', action='store_true', help='Disable LLM stages')
    parser.add_argument('--print-mv', action='store_true', help='Print Marked View')
    parser.add_argument('--reasoning', type=str, default=None, help='Reasoning level (high|medium|low)')
    parser.add_argument('--json', action='store_true', help='Print compact JSON only')
    args = parser.parse_args()

    use_llm = not args.no_llm and os.getenv('BLOCK1_NO_LLM') != '1'

    result = run_full(args.line, use_llm=use_llm, print_mv=args.print_mv, reasoning_level=args.reasoning)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("\n=== RESULT SUMMARY ===")
        print(f"Original:   {result['normalized']['original']}")
        print(f"Normalized: {result['normalized']['text']}")
        print(f"Applied:    {result['normalized']['applied_text']}")
        if result['operations']:
            print(f"Ops:        {result['operations']}")
        if 'refine' in result.get('augment', {}):
            print(f"Canonical terms: {result['augment']['refine'].get('canonical_terms')}")
        if 'hints' in result.get('augment', {}):
            hints = result['augment']['hints']
            print(f"Hints tokens: {len(hints.get('token_hints', []))}")
        print(f"Perf ms:    {result.get('perf', {}).get('total_ms')}\n")
        print("Full JSON:")
        pprint(result, width=140)


if __name__ == '__main__':
    sys.exit(main())
