"""
Unit tests for T1 tokenization module.
"""
import pytest
import sys
from pathlib import Path

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.pipeline.n0_normalize import normalize_n0
from src.pipeline.t1_tokenize import T1Tokenizer, tokenize_t1
from src.common.schemas import Token, TokensResult, NormalizationResult, Pair


class TestT1Tokenizer:
    """Test T1 tokenization functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.tokenizer = T1Tokenizer()
    
    def test_empty_input(self):
        """Test tokenization of empty input."""
        n0 = NormalizationResult(raw_text="", normalized_text="")
        result = self.tokenizer.tokenize_t1(n0)
        assert result.text == ""
        assert result.tokens == []
    
    def test_pair_as_single_token(self):
        """Test that pairs are tokenized as single tokens."""
        n0 = NormalizationResult(
            raw_text="18/0",
            normalized_text="18/0",
            pairs=[Pair(text="18/0", A="18", B="0", sep="/", span=[0, 4])],
            numbers=["18", "0"]
        )
        result = self.tokenizer.tokenize_t1(n0)
        
        assert len(result.tokens) == 1
        token = result.tokens[0]
        assert token.kind == "pair"
        assert token.text == "18/0"
        assert token.meta.A == "18"
        assert token.meta.B == "0"
        assert token.meta.sep == "/"
    
    def test_hebrew_word_tokenization(self):
        """Test Hebrew word tokenization."""
        n0 = NormalizationResult(
            raw_text="שתל",
            normalized_text="שתל"
        )
        result = self.tokenizer.tokenize_t1(n0)
        
        assert len(result.tokens) == 1
        assert result.tokens[0].kind == "word"
        assert result.tokens[0].script == "he"
        assert result.tokens[0].text == "שתל"
    
    def test_english_word_tokenization(self):
        """Test English word tokenization."""
        n0 = NormalizationResult(
            raw_text="implant",
            normalized_text="implant"
        )
        result = self.tokenizer.tokenize_t1(n0)
        
        assert len(result.tokens) == 1
        assert result.tokens[0].kind == "word"
        assert result.tokens[0].script == "en"
    
    def test_number_tokenization(self):
        """Test standalone number tokenization."""
        n0 = NormalizationResult(
            raw_text="14",
            normalized_text="14",
            numbers=["14"]
        )
        result = self.tokenizer.tokenize_t1(n0)
        
        assert len(result.tokens) == 1
        assert result.tokens[0].kind == "number"
        assert result.tokens[0].text == "14"
    
    def test_unit_tokenization(self):
        """Test unit tokenization."""
        n0 = NormalizationResult(
            raw_text="2mm",
            normalized_text="2 mm",
            numbers=["2"],
            units_found=["mm"]
        )
        result = self.tokenizer.tokenize_t1(n0)
        
        # Should have number and unit tokens
        assert len(result.tokens) == 2
        assert result.tokens[0].kind == "number"
        assert result.tokens[0].text == "2"
        assert result.tokens[1].kind == "unit"
        assert result.tokens[1].text == "mm"
    
    def test_word_with_hyphen(self):
        """Test that words with hyphens are kept as single tokens."""
        n0 = NormalizationResult(
            raw_text="multi-unit",
            normalized_text="multi-unit"
        )
        result = self.tokenizer.tokenize_t1(n0)
        
        assert len(result.tokens) == 1
        assert result.tokens[0].text == "multi-unit"
        assert result.tokens[0].kind == "word"
    
    def test_complex_example(self):
        """Test complex tokenization."""
        n0 = normalize_n0("מולטי יוניט שתל 14 18/0")
        result = self.tokenizer.tokenize_t1(n0)
        
        # Check token types
        token_kinds = [t.kind for t in result.tokens]
        assert "word" in token_kinds
        assert "number" in token_kinds
        assert "pair" in token_kinds
        
        # Find and verify pair token
        pair_tokens = [t for t in result.tokens if t.kind == "pair"]
        assert len(pair_tokens) == 1
        assert pair_tokens[0].text == "18/0"
    
    def test_span_accuracy(self):
        """Test accuracy of token spans."""
        text = "שן 36"
        n0 = normalize_n0(text)
        result = self.tokenizer.tokenize_t1(n0)
        
        for token in result.tokens:
            span_text = n0.normalized_text[token.span[0]:token.span[1]]
            assert span_text == token.text
    
    def test_token_coverage(self):
        """Test that all non-whitespace text is covered by tokens."""
        text = "כתר על שן 36"
        n0 = normalize_n0(text)
        result = self.tokenizer.tokenize_t1(n0)
        
        # Reconstruct text from tokens (with spaces)
        reconstructed = []
        for token in result.tokens:
            reconstructed.append(token.text)
        
        # Check all words/numbers are present
        assert "כתר" in reconstructed
        assert "על" in reconstructed
        assert "שן" in reconstructed
        assert "36" in reconstructed
    
    def test_token_indexing(self):
        """Test that tokens are correctly indexed."""
        n0 = normalize_n0("שן 14 16")
        result = self.tokenizer.tokenize_t1(n0)
        
        for i, token in enumerate(result.tokens):
            assert token.idx == i
    
    def test_get_pairs_method(self):
        """Test the get_pairs helper method."""
        n0 = normalize_n0("14-16 ו 18/0")
        result = self.tokenizer.tokenize_t1(n0)
        
        pairs = result.get_pairs()
        assert len(pairs) == 2
        assert all(p.kind == "pair" for p in pairs)
    
    def test_get_numbers_method(self):
        """Test the get_numbers helper method."""
        n0 = normalize_n0("שן 36 עם 14-16")
        result = self.tokenizer.tokenize_t1(n0)
        
        numbers = result.get_numbers()
        # Should only get standalone numbers, not pairs
        assert len(numbers) == 1
        assert numbers[0].text == "36"


class TestConvenienceFunction:
    """Test the convenience function."""
    
    def test_tokenize_t1_function(self):
        """Test the standalone tokenize_t1 function."""
        n0 = normalize_n0("שתל 14")
        result = tokenize_t1(n0)
        
        assert isinstance(result, TokensResult)
        assert len(result.tokens) > 0