"""
T1 - Deterministic tokenization for dental text.
"""
import re
from typing import List, Optional, Tuple
from ..common.schemas import Token, TokenMeta, TokensResult, NormalizationResult


class T1Tokenizer:
    """Deterministic tokenizer for dental text."""
    
    def __init__(self):
        # Regex patterns
        self.hebrew_word = re.compile(r'[\u0590-\u05FF]+(?:[-־][\u0590-\u05FF]+)*')
        self.english_word = re.compile(r'[a-zA-Z]+(?:-[a-zA-Z]+)*')
        self.number = re.compile(r'\d+(?:\.\d+)?')
        self.unit = re.compile(r'mm|°|cm|%')
        self.punct = re.compile(r'[.,;:!?()[\]{}״"\'`]')
        self.whitespace = re.compile(r'\s+')
    
    def tokenize_t1(self, n0: NormalizationResult) -> TokensResult:
        """
        Main tokenization function.
        
        Args:
            n0: NormalizationResult from N0 normalization
            
        Returns:
            TokensResult with tokens
        """
        text = n0.normalized_text
        if not text:
            return TokensResult(text="", tokens=[])
        
        # First, mark pairs to handle them as single tokens
        pair_spans = [(p.span[0], p.span[1], p) for p in n0.pairs]
        pair_spans.sort(key=lambda x: x[0])  # Sort by start position
        
        tokens = []
        token_idx = 0
        pos = 0
        
        while pos < len(text):
            # Check if current position starts a pair
            pair_match = None
            for start, end, pair in pair_spans:
                if pos == start:
                    pair_match = (start, end, pair)
                    break
            
            if pair_match:
                # Handle pair as single token
                start, end, pair = pair_match
                tokens.append(Token(
                    idx=token_idx,
                    text=text[start:end],
                    kind="pair",
                    span=[start, end],
                    script="digit",
                    meta=TokenMeta(A=pair.A, B=pair.B, sep=pair.sep)
                ))
                token_idx += 1
                pos = end
                continue
            
            # Skip whitespace
            ws_match = self.whitespace.match(text, pos)
            if ws_match:
                pos = ws_match.end()
                continue
            
            # Try to match different token types
            token = None
            
            # Check for Hebrew word
            match = self.hebrew_word.match(text, pos)
            if match:
                token = Token(
                    idx=token_idx,
                    text=match.group(0),
                    kind="word",
                    span=[match.start(), match.end()],
                    script="he"
                )
            
            # Check for English word
            if not token:
                match = self.english_word.match(text, pos)
                if match:
                    token = Token(
                        idx=token_idx,
                        text=match.group(0),
                        kind="word",
                        span=[match.start(), match.end()],
                        script="en"
                    )
            
            # Check for number (not part of pair)
            if not token:
                match = self.number.match(text, pos)
                if match:
                    # Verify this number isn't part of a pair
                    is_pair_number = False
                    for start, end, _ in pair_spans:
                        if match.start() >= start and match.end() <= end:
                            is_pair_number = True
                            break
                    
                    if not is_pair_number:
                        token = Token(
                            idx=token_idx,
                            text=match.group(0),
                            kind="number",
                            span=[match.start(), match.end()],
                            script="digit"
                        )
            
            # Check for unit
            if not token:
                match = self.unit.match(text, pos)
                if match:
                    token = Token(
                        idx=token_idx,
                        text=match.group(0),
                        kind="unit",
                        span=[match.start(), match.end()],
                        script=None
                    )
            
            # Check for punctuation
            if not token:
                match = self.punct.match(text, pos)
                if match:
                    token = Token(
                        idx=token_idx,
                        text=match.group(0),
                        kind="punct",
                        span=[match.start(), match.end()],
                        script=None
                    )
            
            # If nothing matched, take single character as unknown
            if not token:
                token = Token(
                    idx=token_idx,
                    text=text[pos],
                    kind="word",  # Default to word
                    span=[pos, pos + 1],
                    script="mixed"
                )
            
            tokens.append(token)
            token_idx += 1
            pos = token.span[1]
        
        return TokensResult(
            text=n0.normalized_text,
            tokens=tokens
        )
    
    def _detect_script(self, text: str) -> str:
        """Detect the script of text."""
        has_hebrew = bool(re.search(r'[\u0590-\u05FF]', text))
        has_english = bool(re.search(r'[a-zA-Z]', text))
        has_digit = bool(re.search(r'\d', text))
        
        if has_hebrew and not has_english:
            return "he"
        elif has_english and not has_hebrew:
            return "en"
        elif has_digit and not has_hebrew and not has_english:
            return "digit"
        else:
            return "mixed"


# Convenience function
def tokenize_t1(n0: NormalizationResult) -> TokensResult:
    """
    Tokenize text using T1 rules.
    
    Args:
        n0: NormalizationResult from N0
        
    Returns:
        TokensResult
    """
    tokenizer = T1Tokenizer()
    return tokenizer.tokenize_t1(n0)