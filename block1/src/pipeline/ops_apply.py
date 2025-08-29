from __future__ import annotations

from typing import List, Tuple, Any
import time
from .t1_tokenize import T1Tokenizer
from .n0_normalize import N0Normalizer
from .validators import SafetyValidator
from ..common.schemas import TokenOperation, TokensResult, NormalizationResult, NormalizationResult as N0Res


def apply_operations(original_text: str, tokens_res: TokensResult, ops: List[TokenOperation]) -> Tuple[str, TokensResult]:
    """Apply insert_space / merge_tokens operations on spacing only.
    Implementation is conservative: only adds/removes single ASCII space boundaries.
    """
    text = original_text
    # Build token boundary positions once
    # For insert_space(after_token_idx): insert a space after end of that token if not present
    # For merge_tokens(range): remove single spaces between consecutive tokens in the range
    if not ops:
        return text, tokens_res
    # Sort ops so that merges happen before inserts inside overlapping regions to stabilize indices
    merges = [o for o in ops if o.op == 'merge_tokens' and o.range]
    inserts = [o for o in ops if o.op == 'insert_space' and o.after_token_idx is not None]
    # Apply merges: remove spaces between tokens start..end
    for op in merges:
        start_idx, end_idx = op.range
        # Iterate over gaps between tokens
        for tidx in range(start_idx, end_idx):
            if tidx < len(tokens_res.tokens) - 1:
                a = tokens_res.tokens[tidx]
                b = tokens_res.tokens[tidx + 1]
                gap_text = text[a.span[1]:b.span[0]]
                if gap_text.strip() == '' and gap_text != '':
                    # remove gap
                    text = text[:a.span[1]] + text[b.span[0]:]
                    shift = len(gap_text)
                    # update following token spans
                    for t2 in tokens_res.tokens[tidx+1:]:
                        t2.span[0] -= shift
                        t2.span[1] -= shift
    # Apply inserts
    for op in inserts:
        idx = op.after_token_idx
        if 0 <= idx < len(tokens_res.tokens):
            tok = tokens_res.tokens[idx]
            # only insert if next char not space
            if tok.span[1] == len(text) or text[tok.span[1]] != ' ':
                text = text[:tok.span[1]] + ' ' + text[tok.span[1]:]
                # shift following tokens
                for t2 in tokens_res.tokens[idx+1:]:
                    t2.span[0] += 1
                    t2.span[1] += 1
    # Re-tokenize quickly using original normalization assumption
    # We reuse T1Tokenizer on a synthetic NormalizationResult shell
    n0_shell = NormalizationResult(
        raw_text=text,
        normalized_text=text,
        numbers=[],
        pairs=[],
        units_found=[],
        dates=[],
        times=[],
        notes=[]
    )
    tokenizer = T1Tokenizer()
    new_tokens = tokenizer.tokenize_t1(n0_shell)
    return text, new_tokens


def orchestrate_block1(line: str, llm_refine_fn=None, llm_hints_fn=None, reasoning_level: str | None = None):
    t0 = time.time()
    normalizer = N0Normalizer()
    n0: NormalizationResult = normalizer.normalize_n0(line)
    tokenizer = T1Tokenizer()
    tokens_res = tokenizer.tokenize_t1(n0)
    augment = {}
    applied_text = n0.normalized_text
    applied_tokens = tokens_res
    ops_objs: List[TokenOperation] = []
    if llm_refine_fn:
        refine = llm_refine_fn(applied_text, applied_tokens)
        augment['refine'] = refine
        ops_raw = refine.get('ops') if isinstance(refine, dict) else []
        for od in ops_raw:
            try:
                ops_objs.append(TokenOperation(**od))
            except Exception:
                continue
        if ops_objs:
            applied_text, applied_tokens = apply_operations(applied_text, applied_tokens, ops_objs)
    if llm_hints_fn:
        hints = llm_hints_fn(applied_text, applied_tokens)
        augment['hints'] = hints
    # Post-ops safety: ensure all original digits still present (multiset compare ignoring spaces)
    def digits(s: str):
        return sorted(ch for ch in s if ch.isdigit())
    if digits(n0.normalized_text) != digits(applied_text):
        augment['post_ops_warning'] = 'digit_mismatch'
    duration = time.time() - t0
    return {
        'normalized': {'original': line, 'text': n0.normalized_text, 'applied_text': applied_text, 'notes': n0.notes},
        'tokens': [t.model_dump() for t in applied_tokens.tokens],
        'operations': [o.model_dump() for o in ops_objs],
        'augment': augment,
        'perf': {'total_ms': round(duration*1000, 2)}
    }
