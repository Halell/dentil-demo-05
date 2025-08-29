"""
Unit tests for LLM invariants and safety validation.
"""
import pytest
import sys
from pathlib import Path

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.common.schemas import (
    LlmAugmentResult, 
    TokenOperation, 
    ToothGroup,
    NormalizationResult,
    TokensResult,
    Token,
    Pair
)
from src.pipeline.validators import SafetyValidator, validate_llm_output


class TestSafetyValidator:
    """Test LLM safety validation."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.validator = SafetyValidator()
        
        # Mock N0 result
        self.n0_result = NormalizationResult(
            raw_text="שתל 14 18/0",
            normalized_text="שתל 14 18/0",
            numbers=["14", "18", "0"],
            pairs=[Pair(text="18/0", A="18", B="0", sep="/", span=[7, 11])]
        )
        
        # Mock T1 result  
        self.t1_result = TokensResult(
            text="שתל 14 18/0",
            tokens=[
                Token(idx=0, text="שתל", kind="word", span=[0, 3], script="he"),
                Token(idx=1, text="14", kind="number", span=[4, 6], script="digit"),
                Token(idx=2, text="18/0", kind="pair", span=[7, 11], script="digit")
            ]
        )
    
    def test_valid_safe_operations(self):
        """Test that safe operations pass validation."""
        llm_result = LlmAugmentResult(
            ops=[],  # No operations - should be safe
            canonical_terms=["dental implant"],
            tooth_groups=[],
            intent_hints=["treatment"],
            ambiguous=False
        )
        
        is_valid, errors = self.validator.validate_llm_output(
            llm_result, self.n0_result, self.t1_result
        )
        
        assert is_valid
        assert len(errors) == 0
    
    def test_reject_pair_modification(self):
        """Test that pair modification is rejected."""
        # Try to insert space after pair token
        llm_result = LlmAugmentResult(
            ops=[TokenOperation(op="insert_space", after_token_idx=2)],  # Token 2 is pair
            canonical_terms=[],
            tooth_groups=[],
            intent_hints=[],
            ambiguous=False
        )
        
        is_valid, errors = self.validator.validate_llm_output(
            llm_result, self.n0_result, self.t1_result
        )
        
        assert not is_valid
        assert any("pair token" in error for error in errors)
    
    def test_reject_number_modification(self):
        """Test that number token modification is rejected."""
        # Try to merge tokens including number
        llm_result = LlmAugmentResult(
            ops=[TokenOperation(op="merge_tokens", range=[1, 2])],  # Includes number token
            canonical_terms=[],
            tooth_groups=[],
            intent_hints=[],
            ambiguous=False
        )
        
        is_valid, errors = self.validator.validate_llm_output(
            llm_result, self.n0_result, self.t1_result
        )
        
        assert not is_valid
        assert len(errors) > 0
    
    def test_invalid_tooth_groups(self):
        """Test rejection of invalid tooth groups."""
        llm_result = LlmAugmentResult(
            ops=[],
            canonical_terms=[],
            tooth_groups=[
                ToothGroup(
                    label_he="קבוצה בדויה",
                    label_en="fake group",
                    FDI=["99", "100"]  # Invalid FDI numbers
                )
            ],
            intent_hints=[],
            ambiguous=False
        )
        
        is_valid, errors = self.validator.validate_llm_output(
            llm_result, self.n0_result, self.t1_result
        )
        
        assert not is_valid
        assert any("Invalid FDI" in error for error in errors)
    
    def test_valid_tooth_groups(self):
        """Test acceptance of valid tooth groups."""
        llm_result = LlmAugmentResult(
            ops=[],
            canonical_terms=[],
            tooth_groups=[
                ToothGroup(
                    label_he="חותכות תחתונות",
                    label_en="lower incisors",
                    FDI=["31", "32", "41", "42"]  # Valid FDI numbers
                )
            ],
            intent_hints=[],
            ambiguous=False
        )
        
        is_valid, errors = self.validator.validate_llm_output(
            llm_result, self.n0_result, self.t1_result
        )
        
        assert is_valid
        assert len(errors) == 0
    
    def test_invalid_operation_parameters(self):
        """Test rejection of invalid operation parameters."""
        # Out of range token index
        llm_result = LlmAugmentResult(
            ops=[TokenOperation(op="insert_space", after_token_idx=99)],  # Out of range
            canonical_terms=[],
            tooth_groups=[],
            intent_hints=[],
            ambiguous=False
        )
        
        is_valid, errors = self.validator.validate_llm_output(
            llm_result, self.n0_result, self.t1_result
        )
        
        assert not is_valid
        assert any("out of range" in error for error in errors)
    
    def test_invalid_merge_range(self):
        """Test rejection of invalid merge ranges."""
        llm_result = LlmAugmentResult(
            ops=[TokenOperation(op="merge_tokens", range=[2, 1])],  # Invalid range
            canonical_terms=[],
            tooth_groups=[],
            intent_hints=[],
            ambiguous=False
        )
        
        is_valid, errors = self.validator.validate_llm_output(
            llm_result, self.n0_result, self.t1_result
        )
        
        assert not is_valid
        assert any("Invalid merge range" in error for error in errors)
    
    def test_word_only_operations_allowed(self):
        """Test that operations on word tokens are allowed."""
        # Create result with only word tokens
        word_only_t1 = TokensResult(
            text="מילה אחרת",
            tokens=[
                Token(idx=0, text="מילה", kind="word", span=[0, 4], script="he"),
                Token(idx=1, text="אחרת", kind="word", span=[5, 9], script="he")
            ]
        )
        
        word_only_n0 = NormalizationResult(
            raw_text="מילה אחרת",
            normalized_text="מילה אחרת",
            numbers=[],
            pairs=[]
        )
        
        llm_result = LlmAugmentResult(
            ops=[TokenOperation(op="merge_tokens", range=[0, 1])],  # Merge word tokens
            canonical_terms=[],
            tooth_groups=[],
            intent_hints=[],
            ambiguous=False
        )
        
        is_valid, errors = self.validator.validate_llm_output(
            llm_result, word_only_n0, word_only_t1
        )
        
        assert is_valid
        assert len(errors) == 0


class TestConvenienceFunction:
    """Test the convenience validation function."""
    
    def test_validate_llm_output_function(self):
        """Test standalone validation function."""
        n0 = NormalizationResult(
            raw_text="test",
            normalized_text="test",
            numbers=[],
            pairs=[]
        )
        
        t1 = TokensResult(
            text="test",
            tokens=[Token(idx=0, text="test", kind="word", span=[0, 4], script="en")]
        )
        
        llm_result = LlmAugmentResult(
            ops=[],
            canonical_terms=[],
            tooth_groups=[],
            intent_hints=[],
            ambiguous=False
        )
        
        is_valid, errors = validate_llm_output(llm_result, n0, t1)
        assert is_valid
        assert len(errors) == 0


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_results(self):
        """Test validation with empty results."""
        n0 = NormalizationResult(raw_text="", normalized_text="")
        t1 = TokensResult(text="", tokens=[])
        llm_result = LlmAugmentResult()
        
        validator = SafetyValidator()
        is_valid, errors = validator.validate_llm_output(llm_result, n0, t1)
        
        assert is_valid  # Empty should be valid
    
    def test_no_tooth_groups_dict(self):
        """Test validator without tooth groups dictionary."""
        validator = SafetyValidator(tooth_groups_path="nonexistent.json")
        
        n0 = NormalizationResult(raw_text="test", normalized_text="test")
        t1 = TokensResult(text="test", tokens=[])
        llm_result = LlmAugmentResult()
        
        is_valid, errors = validator.validate_llm_output(llm_result, n0, t1)
        assert is_valid  # Should work without dict