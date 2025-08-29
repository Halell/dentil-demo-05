"""
N0b - LLM-based refinement for normalization.
Safe, controlled refinements only.
"""
import json
from typing import Dict, Optional, List, Any
from pathlib import Path
from ..common.schemas import LlmAugmentResult, TokenOperation, ToothGroup
from ..llm.ollama_client import chat_json
from .validators import validate_llm_output


class N0bRefiner:
    """LLM-based refinement for normalized text."""
    
    def __init__(self, prompt_path: Optional[str] = None):
        """
        Initialize refiner.
        
        Args:
            prompt_path: Path to prompt template file
        """
        if prompt_path:
            self.prompt_template = self._load_prompt(prompt_path)
        else:
            self.prompt_template = self._get_default_prompt()
        
        # Load tooth groups dictionary for validation
        base_path = Path(__file__).parent.parent.parent
        dict_path = base_path / "data" / "dictionaries" / "tooth_groups.json"
        if dict_path.exists():
            with open(dict_path, 'r', encoding='utf-8') as f:
                self.tooth_groups = json.load(f)
        else:
            self.tooth_groups = {}
    
    def _load_prompt(self, path: str) -> str:
        """Load prompt template from file."""
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _get_default_prompt(self) -> str:
        """Get default prompt template."""
        return """You are a dental text analysis assistant. 
Analyze the provided Marked View and suggest SAFE refinements only.

CRITICAL RULES:
1. NEVER modify numbers or number pairs
2. NEVER change pair separators
3. Only suggest operations on words (split/merge)
4. Tooth groups must use exact FDI numbers from the dictionary
5. Return valid JSON only

Expected JSON structure:
{
  "ops": [
    {"op": "insert_space", "after_token_idx": <number>},
    {"op": "merge_tokens", "range": [<start>, <end>]}
  ],
  "canonical_terms": ["term1", "term2"],
  "tooth_groups": [
    {"label_he": "...", "label_en": "...", "FDI": ["31", "32", ...]}
  ],
  "intent_hints": ["scheduling", "treatment", etc],
  "ambiguous": false
}

Analyze the following Marked View:"""
    
    def refine_with_llm(
        self,
        mv_text: str,
        n0_result: Any = None,
        t1_result: Any = None,
        validate: bool = True
    ) -> Dict:
        """
        Refine normalization using LLM.
        
        Args:
            mv_text: Marked View text
            n0_result: Original N0 result (for validation)
            t1_result: Original T1 result (for validation)
            validate: Whether to validate LLM output
            
        Returns:
            Dict with refinement suggestions or error
        """
        # Build prompt
        full_prompt = f"{self.prompt_template}\n\n{mv_text}"
        
        # Call LLM
        try:
            llm_response = chat_json(
                prompt=full_prompt,
                temperature=0.1,  # Low temperature for consistency
                json_only=True
            )
        except Exception as e:
            return {
                "error": f"LLM call failed: {str(e)}",
                "ambiguous": True
            }
        
        # Check for error in response
        if "error" in llm_response:
            return {
                "error": llm_response["error"],
                "ambiguous": True
            }
        
        # Parse into schema
        try:
            # Extract operations
            ops = []
            for op_data in llm_response.get("ops", []):
                if op_data.get("op") == "insert_space":
                    ops.append(TokenOperation(
                        op="insert_space",
                        after_token_idx=op_data.get("after_token_idx")
                    ))
                elif op_data.get("op") == "merge_tokens":
                    ops.append(TokenOperation(
                        op="merge_tokens",
                        range=op_data.get("range")
                    ))
            
            # Extract tooth groups
            tooth_groups = []
            for tg_data in llm_response.get("tooth_groups", []):
                tooth_groups.append(ToothGroup(
                    label_he=tg_data.get("label_he", ""),
                    label_en=tg_data.get("label_en", ""),
                    FDI=tg_data.get("FDI", [])
                ))
            
            # Canonical terms cleanup: remove any containing digits and deduplicate preserving order
            raw_terms = llm_response.get("canonical_terms", [])
            seen = set()
            cleaned_terms = []
            for t in raw_terms:
                if any(ch.isdigit() for ch in t):
                    continue
                low = t.strip()
                if not low:
                    continue
                if low.lower() in seen:
                    continue
                seen.add(low.lower())
                cleaned_terms.append(low)

            # Create result
            result = LlmAugmentResult(
                ops=ops,
                canonical_terms=cleaned_terms,
                tooth_groups=tooth_groups,
                intent_hints=llm_response.get("intent_hints", []),
                ambiguous=llm_response.get("ambiguous", False)
            )
            
            # Validate if requested and inputs provided
            if validate and n0_result and t1_result:
                is_valid, errors = validate_llm_output(result, n0_result, t1_result)
                if not is_valid:
                    return {
                        "error": "Validation failed",
                        "validation_errors": errors,
                        "ambiguous": True
                    }
            
            return result.model_dump()
            
        except Exception as e:
            return {
                "error": f"Failed to parse LLM response: {str(e)}",
                "raw_response": llm_response,
                "ambiguous": True
            }
    
    def apply_operations(
        self,
        tokens: List[Any],
        operations: List[TokenOperation]
    ) -> List[Any]:
        """
        Apply safe operations to tokens.
        
        Args:
            tokens: List of tokens
            operations: List of operations to apply
            
        Returns:
            Modified token list
        """
        # This is a simplified implementation
        # In production, would need careful handling of indices
        result = tokens.copy()
        
        # Sort operations by position (reverse for safety)
        sorted_ops = sorted(
            operations,
            key=lambda x: x.after_token_idx if x.op == "insert_space" else x.range[0],
            reverse=True
        )
        
        for op in sorted_ops:
            if op.op == "insert_space" and op.after_token_idx is not None:
                # Would insert space marker after token
                # This is conceptual - actual implementation would modify text
                pass
            elif op.op == "merge_tokens" and op.range:
                # Would merge tokens in range
                # This is conceptual - actual implementation would combine tokens
                pass
        
        return result


def refine_with_llm(text: str, tokens_result: Any = None):
    """Wrapper for orchestrator: build minimal MV context or use raw text if MV not built."""
    refiner = N0bRefiner()
    mv_text = text  # fallback simple
    return refiner.refine_with_llm(mv_text, None, tokens_result)