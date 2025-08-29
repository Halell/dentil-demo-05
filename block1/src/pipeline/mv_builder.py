"""
MV - Marked View builder for LLM processing.
Creates a structured document for LLM to process safely.
"""
from typing import Optional
from ..common.schemas import NormalizationResult, TokensResult


class MVBuilder:
    """Builds Marked View for LLM processing."""
    
    def build_marked_view(
        self,
        n0: NormalizationResult,
        t1: TokensResult,
        include_example: bool = True,
        reasoning_level: str | None = None
    ) -> str:
        """Build a Marked View document for LLM."""
        sections: list[str] = []
        sections.append(self._build_legend(reasoning_level))
        sections.append(self._build_text_section(n0))
        sections.append(self._build_tokens_section(t1))
        if n0.pairs:
            sections.append(self._build_pairs_section(n0))
        if n0.numbers:
            sections.append(self._build_numbers_section(n0))
        sections.append(self._build_safety_rules())
        sections.append(self._build_instructions())
        if include_example:
            sections.append(self._build_example())
        return "\n\n".join(sections)
    
    def _build_legend(self, reasoning_level: str | None) -> str:
        """Build legend section."""
        rl = f"Reasoning: {reasoning_level}" if reasoning_level else ""
        return f"""<LEGEND>
Token Types:
- word: Hebrew or English word (may include hyphens)
- number: Standalone number
- pair: Number pair (e.g., 18/0, 14-16)
- unit: Measurement unit (mm, °, etc.)
- punct: Punctuation mark

{rl}
Safety Rules:
- NEVER modify numbers or number pairs
- NEVER change pair separators (/, \\, -)
- Only suggest safe word operations
- Return JSON only, no free text
</LEGEND>"""
    
    def _build_text_section(self, n0: NormalizationResult) -> str:
        """Build text section."""
        return f"""<TEXT>
Original: {n0.raw_text}
Normalized: {n0.normalized_text}
</TEXT>"""
    
    def _build_tokens_section(self, t1: TokensResult) -> str:
        """Build tokens section."""
        lines = ["<TOKENS>"]
        for token in t1.tokens:
            meta_str = ""
            if token.meta:
                meta_str = f" [A={token.meta.A}, B={token.meta.B}, sep={token.meta.sep}]"
            
            lines.append(
                f"[{token.idx}] {token.text} "
                f"({token.kind}, span={token.span}, script={token.script}){meta_str}"
            )
        lines.append("</TOKENS>")
        return "\n".join(lines)
    
    def _build_pairs_section(self, n0: NormalizationResult) -> str:
        """Build pairs section."""
        lines = ["<PAIRS>"]
        for pair in n0.pairs:
            lines.append(
                f"- {pair.text}: A={pair.A}, B={pair.B}, "
                f"separator='{pair.sep}', span={pair.span}"
            )
        lines.append("</PAIRS>")
        return "\n".join(lines)
    
    def _build_numbers_section(self, n0: NormalizationResult) -> str:
        """Build numbers section."""
        return f"""<NUMBERS>
Found numbers: {', '.join(n0.numbers)}
Count: {len(n0.numbers)}
</NUMBERS>"""
    
    def _build_safety_rules(self) -> str:
        """Build safety rules section."""
        return """<SAFETY_RULES>
CRITICAL - These are invariants that MUST be maintained:
1. The list of numbers must remain EXACTLY the same (order and values)
2. All pairs must remain unchanged (same A, B, and separator)
3. No characters can be lost (only word splitting/merging allowed)
4. Tooth groups must only use FDI numbers from the approved dictionary
5. If uncertain, set ambiguous=true
</SAFETY_RULES>"""
    
    def _build_instructions(self) -> str:
        """Build instructions section."""
        return """<INSTRUCTIONS>
Analyze the dental text and return a JSON object with:

1. "ops": List of safe operations on words only:
   - {"op": "insert_space", "after_token_idx": N}
   - {"op": "merge_tokens", "range": [start, end]}

2. "canonical_terms": English dental terms for the concepts

3. "tooth_groups": Identified tooth groups with:
   - "label_he": Hebrew name
   - "label_en": English name  
   - "FDI": List of tooth numbers (MUST be from dictionary)

4. "intent_hints": Scheduling, treatment, examination, etc.

5. "ambiguous": true if text is unclear or uncertain

Return ONLY valid JSON. No explanations or comments.
</INSTRUCTIONS>"""
    
    def _build_example(self) -> str:
        """Build example section."""
        return """<EXAMPLE>
Input: "מולטי יוניט שתל 14 18/0"
Output:
{
  "ops": [],
  "canonical_terms": ["multi-unit abutment", "dental implant"],
  "tooth_groups": [],
  "intent_hints": ["treatment"],
  "ambiguous": false
}
</EXAMPLE>"""


def build_marked_view(
    n0: NormalizationResult,
    t1: TokensResult,
    include_example: bool = True
) -> str:
    """
    Build Marked View for LLM processing.
    
    Args:
        n0: NormalizationResult
        t1: TokensResult
        include_example: Whether to include example
        
    Returns:
        Marked View string
    """
    builder = MVBuilder()
    return builder.build_marked_view(n0, t1, include_example)