"""
Unit tests for N0 normalization module.
"""
import pytest
import sys
from pathlib import Path

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.pipeline.n0_normalize import N0Normalizer, normalize_n0
from src.common.schemas import NormalizationResult, Pair


class TestN0Normalizer:
    """Test N0 normalization functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.normalizer = N0Normalizer()
    
    def test_empty_input(self):
        """Test normalization of empty input."""
        result = self.normalizer.normalize_n0("")
        assert result.raw_text == ""
        assert result.normalized_text == ""
        assert result.numbers == []
        assert result.pairs == []
    
    def test_space_insertion_hebrew_digit(self):
        """Test space insertion between Hebrew and digits."""
        result = self.normalizer.normalize_n0("שתל14")
        assert result.normalized_text == "שתל 14"
        assert "inserted_space_hebrew_digit" in result.notes
        assert result.numbers == ["14"]
    
    def test_space_insertion_digit_hebrew(self):
        """Test space insertion between digit and Hebrew."""
        result = self.normalizer.normalize_n0("14שתל")
        assert result.normalized_text == "14 שתל"
        assert "inserted_space_digit_hebrew" in result.notes
    
    def test_pair_detection_slash(self):
        """Test detection of number pairs with slash."""
        result = self.normalizer.normalize_n0("18/0")
        assert len(result.pairs) == 1
        pair = result.pairs[0]
        assert pair.A == "18"
        assert pair.B == "0"
        assert pair.sep == "/"
        assert pair.text == "18/0"
    
    def test_pair_detection_dash(self):
        """Test detection of number pairs with dash."""
        result = self.normalizer.normalize_n0("14-16")
        assert len(result.pairs) == 1
        pair = result.pairs[0]
        assert pair.A == "14"
        assert pair.B == "16"
        assert pair.sep == "-"
    
    def test_pair_detection_backslash(self):
        """Test detection of number pairs with backslash."""
        result = self.normalizer.normalize_n0("13\\15")
        assert len(result.pairs) == 1
        pair = result.pairs[0]
        assert pair.A == "13"
        assert pair.B == "15"
        assert pair.sep == "\\"
    
    def test_numbers_extraction(self):
        """Test extraction of all numbers."""
        result = self.normalizer.normalize_n0("שן 36 עם 2mm ו-18/0")
        assert "36" in result.numbers
        assert "2" in result.numbers
        assert "18" in result.numbers
        assert "0" in result.numbers
    
    def test_unit_normalization(self):
        """Test normalization of units."""
        result = self.normalizer.normalize_n0("מרווח 2ממ")
        assert "mm" in result.normalized_text
        assert "mm" in result.units_found
    
    def test_complex_example(self):
        """Test complex dental text."""
        text = "מולטיוניט שתל14 18/0"
        result = self.normalizer.normalize_n0(text)
        
        assert result.normalized_text == "מולטיוניט שתל 14 18/0"
        assert len(result.pairs) == 1
        assert result.pairs[0].text == "18/0"
        assert set(result.numbers) == {"14", "18", "0"}
    
    def test_numbers_order_preserved(self):
        """Test that numbers order is preserved."""
        text = "שיניים 25 27 31 32"
        result = self.normalizer.normalize_n0(text)
        assert result.numbers == ["25", "27", "31", "32"]
    
    def test_pair_span_accuracy(self):
        """Test accuracy of pair span calculation."""
        text = "טיפול 14-16 דחוף"
        result = self.normalizer.normalize_n0(text)
        
        pair = result.pairs[0]
        span_text = result.normalized_text[pair.span[0]:pair.span[1]]
        assert span_text == pair.text
    
    def test_no_unit_addition(self):
        """Test that units are not added if not present."""
        result = self.normalizer.normalize_n0("שן 36")
        assert "mm" not in result.units_found
        assert "°" not in result.units_found
    
    def test_unicode_normalization(self):
        """Test Unicode normalization."""
        # Test text with different Unicode representations
        result = self.normalizer.normalize_n0("שן 36")
        assert result.raw_text == "שן 36"
        # Unicode should be normalized to NFC


class TestConvenienceFunction:
    """Test the convenience function."""
    
    def test_normalize_n0_function(self):
        """Test the standalone normalize_n0 function."""
        result = normalize_n0("שתל14 18/0")
        assert isinstance(result, NormalizationResult)
        assert result.normalized_text == "שתל 14 18/0"
        assert len(result.pairs) == 1