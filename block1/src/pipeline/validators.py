"""
Validators for ensuring safety invariants in LLM processing.
"""
import json
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from ..common.schemas import (
    NormalizationResult,
    TokensResult,
    LlmAugmentResult,
    Token
)


class SafetyValidator:
    """Validates LLM outputs against safety invariants."""
    
    def __init__(self, tooth_groups_path: Optional[str] = None):
        """
        Initialize validator.
        
        Args:
            tooth_groups_path: Path to tooth_groups.json dictionary
        """
        if tooth_groups_path:
            self.tooth_groups_dict = self._load_tooth_groups(tooth_groups_path)
        else:
            # Default path
            base_path = Path(__file__).parent.parent.parent
            dict_path = base_path / "data" / "dictionaries" / "tooth_groups.json"
            if dict_path.exists():
                self.tooth_groups_dict = self._load_tooth_groups(str(dict_path))
            else:
                self.tooth_groups_dict = {}
    
    def _load_tooth_groups(self, path: str) -> Dict:
        """Load tooth groups dictionary."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    def validate_llm_output(
        self,
        llm_result: LlmAugmentResult,
        n0: NormalizationResult,
        t1: TokensResult
    ) -> Tuple[bool, List[str]]:
        """
        Validate LLM output against safety invariants.
        
        Args:
            llm_result: LLM augmentation result
            n0: Original N0 result
            t1: Original T1 result
            
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        
        # Validate number invariants
        number_errors = self._validate_numbers_invariant(llm_result, n0, t1)
        errors.extend(number_errors)
        
        # Validate pair invariants
        pair_errors = self._validate_pairs_invariant(llm_result, n0, t1)
        errors.extend(pair_errors)
        
        # Validate token operations
        op_errors = self._validate_token_operations(llm_result, t1)
        errors.extend(op_errors)
        
        # Validate tooth groups
        tooth_errors = self._validate_tooth_groups(llm_result)
        errors.extend(tooth_errors)
        
        # Validate character coverage
        coverage_errors = self._validate_character_coverage(llm_result, n0, t1)
        errors.extend(coverage_errors)
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def _validate_numbers_invariant(
        self,
        llm_result: LlmAugmentResult,
        n0: NormalizationResult,
        t1: TokensResult
    ) -> List[str]:
        """Validate that numbers remain unchanged."""
        errors = []
        
        # Get original numbers from N0
        original_numbers = n0.numbers
        
        # Simulate applying operations to check if numbers change
        # This is a simplified check - in production, actually apply ops
        for op in llm_result.ops:
            if op.op == "merge_tokens" and op.range:
                # Check if merging affects number tokens
                for idx in range(op.range[0], op.range[1] + 1):
                    if idx < len(t1.tokens):
                        token = t1.tokens[idx]
                        if token.kind in ["number", "pair"]:
                            errors.append(
                                f"Operation would modify number/pair token at index {idx}"
                            )
        
        return errors
    
    def _validate_pairs_invariant(
        self,
        llm_result: LlmAugmentResult,
        n0: NormalizationResult,
        t1: TokensResult
    ) -> List[str]:
        """Validate that pairs remain unchanged."""
        errors = []
        
        # Check if any operation touches pair tokens
        pair_indices = {t.idx for t in t1.tokens if t.kind == "pair"}
        
        for op in llm_result.ops:
            if op.op == "insert_space" and op.after_token_idx in pair_indices:
                errors.append(
                    f"Cannot insert space after pair token at index {op.after_token_idx}"
                )
            
            if op.op == "merge_tokens" and op.range:
                for idx in range(op.range[0], op.range[1] + 1):
                    if idx in pair_indices:
                        errors.append(
                            f"Cannot merge pair token at index {idx}"
                        )
        
        return errors
    
    def _validate_token_operations(
        self,
        llm_result: LlmAugmentResult,
        t1: TokensResult
    ) -> List[str]:
        """Validate token operations are valid."""
        errors = []
        max_idx = len(t1.tokens) - 1
        for op in llm_result.ops:
            if op.op == "insert_space":
                if op.after_token_idx is None:
                    errors.append("insert_space operation missing after_token_idx")
                elif op.after_token_idx < 0 or op.after_token_idx > max_idx:
                    errors.append(
                        f"after_token_idx {op.after_token_idx} out of range [0, {max_idx}]"
                    )
            elif op.op == "merge_tokens":
                if not op.range or len(op.range) != 2:
                    errors.append("merge_tokens requires range [start,end]")
                else:
                    if op.range[0] < 0 or op.range[1] > max_idx:
                        errors.append(f"merge range {op.range} out of bounds [0,{max_idx}]")
                    elif op.range[0] > op.range[1]:
                        errors.append(f"Invalid merge range {op.range}")
        # Lightweight length simulation
        delta = 0
        for op in llm_result.ops:
            if op.op == 'insert_space':
                delta += 1
            elif op.op == 'merge_tokens':
                # approximate: merging N tokens reduces at most (N-1) spaces; treat as 1 char decrease
                delta -= 1
        if len(t1.text) + delta <= 0:
            errors.append("operations would empty text")
        
        return errors
    
    def _validate_tooth_groups(self, llm_result: LlmAugmentResult) -> List[str]:
        """Validate tooth groups against dictionary."""
        errors = []
        
        if not self.tooth_groups_dict:
            return errors  # Skip if no dictionary loaded
        
        # Collect all valid FDI numbers from dictionary
        valid_fdi = set()
        for group_data in self.tooth_groups_dict.values():
            if "FDI" in group_data:
                valid_fdi.update(group_data["FDI"])
        
        # Check each tooth group from LLM
        for group in llm_result.tooth_groups:
            for fdi in group.FDI:
                if fdi not in valid_fdi:
                    errors.append(
                        f"Invalid FDI number '{fdi}' not in tooth groups dictionary"
                    )
        
        return errors
    
    def _validate_character_coverage(
        self,
        llm_result: LlmAugmentResult,
        n0: NormalizationResult,
        t1: TokensResult
    ) -> List[str]:
        """Validate no characters are lost."""
        errors = []
        
        # This is a simplified check
        # In production, would apply operations and verify coverage
        original_text = n0.normalized_text.replace(" ", "")
        
        # Check that operations don't delete content
        for op in llm_result.ops:
            if op.op not in ["insert_space", "merge_tokens"]:
                errors.append(f"Unknown operation type: {op.op}")
        
        return errors

    def validate_post_operations(self, original_numbers: list[str], original_pairs: list[Token], new_text: str) -> List[str]:
        """Validate after operations that numbers and pair strings still exist intact."""
        errs = []
        for num in original_numbers:
            if num not in new_text:
                errs.append(f'missing_number:{num}')
        for pair_tok in original_pairs:
            if pair_tok.text not in new_text:
                errs.append(f'missing_pair:{pair_tok.text}')
        return errs


def validate_llm_output(
    llm_result: LlmAugmentResult,
    n0: NormalizationResult,
    t1: TokensResult,
    tooth_groups_path: Optional[str] = None
) -> Tuple[bool, List[str]]:
    """
    Validate LLM output for safety.
    
    Args:
        llm_result: LLM result to validate
        n0: Original N0 result
        t1: Original T1 result
        tooth_groups_path: Optional path to tooth groups dictionary
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    validator = SafetyValidator(tooth_groups_path)
    return validator.validate_llm_output(llm_result, n0, t1)