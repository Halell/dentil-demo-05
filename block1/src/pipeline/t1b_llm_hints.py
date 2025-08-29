"""
T1b - LLM-based hints and tags for tokens.
Adds semantic hints without modifying tokens.
"""
import json
from typing import Dict, List, Optional, Any
from pathlib import Path
from ..llm.ollama_client import chat_json


class T1bHintGenerator:
    """Generate hints and tags for tokens using LLM."""
    
    def __init__(self):
        """Initialize hint generator."""
        # Load tooth groups for validation
        base_path = Path(__file__).parent.parent.parent
        dict_path = base_path / "data" / "dictionaries" / "tooth_groups.json"
        if dict_path.exists():
            with open(dict_path, 'r', encoding='utf-8') as f:
                self.tooth_groups = json.load(f)
        else:
            self.tooth_groups = {}
    
    def generate_hints(
        self,
        mv_text: str,
        tokens: List[Any]
    ) -> Dict:
        """
        Generate hints for tokens.
        
        Args:
            mv_text: Marked View text
            tokens: List of tokens from T1
            
        Returns:
            Dict with hints for tokens
        """
        prompt = self._build_prompt(mv_text, tokens)
        
        try:
            llm_response = chat_json(
                prompt=prompt,
                system=self._get_system_prompt(),
                temperature=0.2,
                json_only=True
            )
        except Exception as e:
            return {
                "error": f"LLM call failed: {str(e)}",
                "token_hints": []
            }
        
        # Parse and validate hints
        return self._parse_hints(llm_response, tokens)
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for hint generation."""
        return """You are a dental terminology expert.
Analyze tokens and provide semantic hints.

RULES:
1. DO NOT modify tokens or their text
2. Only add semantic labels/hints
3. Identify device types (implant, crown, etc)
4. Identify tooth groups from the approved dictionary
5. Suggest intent (treatment, scheduling, examination)

Return JSON with:
{
  "token_hints": [
    {
      "idx": <token_index>,
      "device_hint": "implant|crown|bridge|etc",
      "tooth_group": "upper_molars|lower_incisors|etc",
      "clinical_term": "canonical English term"
    }
  ],
  "overall_intent": ["treatment", "scheduling", etc],
  "detected_procedures": ["root_canal", "crown_placement", etc]
}"""
    
    def _build_prompt(self, mv_text: str, tokens: List[Any]) -> str:
        """Build prompt for LLM."""
        # Extract token info
        token_summary = []
        for t in tokens[:20]:  # Limit to first 20 for context
            token_summary.append(f"[{t.idx}] {t.text} ({t.kind})")
        
        return f"""Analyze these dental tokens and provide hints:

Marked View:
{mv_text}

Token List:
{chr(10).join(token_summary)}

Provide semantic hints for relevant tokens."""
    
    def _parse_hints(self, llm_response: Dict, tokens: List[Any]) -> Dict:
        """Parse and validate hints from LLM."""
        if "error" in llm_response:
            return {"error": llm_response["error"], "token_hints": []}
        
        result = {
            "token_hints": [],
            "overall_intent": llm_response.get("overall_intent", []),
            "detected_procedures": llm_response.get("detected_procedures", [])
        }
        
        # Process token hints
        for hint_data in llm_response.get("token_hints", []):
            idx = hint_data.get("idx")
            if idx is None or idx >= len(tokens):
                continue
            
            hint = {
                "idx": idx,
                "token_text": tokens[idx].text if idx < len(tokens) else ""
            }
            
            # Add device hint
            if "device_hint" in hint_data:
                hint["device_hint"] = hint_data["device_hint"]
            
            # Add tooth group if valid
            tooth_group = hint_data.get("tooth_group")
            if tooth_group and tooth_group in self.tooth_groups:
                hint["tooth_group"] = tooth_group
                hint["tooth_group_fdi"] = self.tooth_groups[tooth_group]["FDI"]
            
            # Add clinical term
            if "clinical_term" in hint_data:
                hint["clinical_term"] = hint_data["clinical_term"]
            
            result["token_hints"].append(hint)
        
        return result
    
    def enrich_tokens(
        self,
        tokens: List[Any],
        hints: Dict
    ) -> List[Any]:
        """
        Enrich tokens with hints (non-destructive).
        
        Args:
            tokens: Original tokens
            hints: Hints from LLM
            
        Returns:
            Enriched tokens (copies with hints added)
        """
        enriched = []
        hint_map = {h["idx"]: h for h in hints.get("token_hints", [])}
        
        for token in tokens:
            # Create a copy (assuming token has a dict representation)
            enriched_token = token.model_dump() if hasattr(token, 'model_dump') else dict(token)
            
            # Add hints if available
            if token.idx in hint_map:
                enriched_token["hints"] = hint_map[token.idx]
            
            enriched.append(enriched_token)
        
        return enriched


def generate_token_hints(
    mv_text: str,
    tokens: List[Any]
) -> Dict:
    """
    Generate hints for tokens.
    
    Args:
        mv_text: Marked View
        tokens: Token list
        
    Returns:
        Hints dictionary
    """
    generator = T1bHintGenerator()
    return generator.generate_hints(mv_text, tokens)


def generate_hints_with_llm(text: str, tokens_result: Any = None):
    """Wrapper for orchestrator expecting plain text and TokensResult"""
    tokens = tokens_result.tokens if tokens_result else []
    return generate_token_hints(text, tokens)