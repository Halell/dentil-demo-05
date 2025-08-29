"""
Pydantic schemas for Block 1 data structures.
"""
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field


class Pair(BaseModel):
    """Represents a number pair (e.g., 18/0, 14-16)."""
    text: str = Field(..., description="The full pair text")
    A: str = Field(..., description="First number")
    B: str = Field(..., description="Second number")
    sep: Literal["/", "\\", "-"] = Field(..., description="Separator character")
    span: List[int] = Field(..., min_items=2, max_items=2, description="Start and end positions")


class NormalizationResult(BaseModel):
    """Result from N0 normalization step."""
    raw_text: str = Field(..., description="Original input text")
    normalized_text: str = Field(..., description="Normalized text")
    numbers: List[str] = Field(default_factory=list, description="All numbers found")
    pairs: List[Pair] = Field(default_factory=list, description="Number pairs detected")
    units_found: List[str] = Field(default_factory=list, description="Units found in text")
    dates: List[str] = Field(default_factory=list, description="Date patterns found")
    times: List[str] = Field(default_factory=list, description="Time patterns found")
    notes: List[str] = Field(default_factory=list, description="Processing notes")


class TokenMeta(BaseModel):
    """Metadata for special tokens."""
    A: Optional[str] = None
    B: Optional[str] = None
    sep: Optional[str] = None


class Token(BaseModel):
    """Single token from T1 tokenization."""
    idx: int = Field(..., ge=0, description="Token index")
    text: str = Field(..., description="Token text")
    kind: Literal["word", "number", "pair", "unit", "punct"] = Field(..., description="Token type")
    span: List[int] = Field(..., min_items=2, max_items=2, description="Start and end positions")
    script: Optional[Literal["he", "en", "digit", "mixed"]] = Field(None, description="Script type")
    meta: Optional[TokenMeta] = Field(None, description="Metadata for pairs")


class TokensResult(BaseModel):
    """Result from T1 tokenization step."""
    text: str = Field(..., description="Input text")
    tokens: List[Token] = Field(..., description="List of tokens")
    
    def get_pairs(self) -> List[Token]:
        """Get all pair tokens."""
        return [t for t in self.tokens if t.kind == "pair"]
    
    def get_numbers(self) -> List[Token]:
        """Get all number tokens (excluding pairs)."""
        return [t for t in self.tokens if t.kind == "number"]


class TokenOperation(BaseModel):
    """Operation to apply on tokens."""
    op: Literal["insert_space", "merge_tokens"] = Field(..., description="Operation type")
    after_token_idx: Optional[int] = Field(None, description="Insert space after this token")
    range: Optional[List[int]] = Field(None, description="Token range for merge [start, end]")


class ToothGroup(BaseModel):
    """Tooth group information."""
    label_he: str = Field(..., description="Hebrew label")
    label_en: str = Field(..., description="English label")
    FDI: List[str] = Field(..., description="FDI tooth numbers")


class LlmAugmentResult(BaseModel):
    """Result from LLM augmentation (N0b/T1b)."""
    ops: List[TokenOperation] = Field(default_factory=list, description="Token operations")
    canonical_terms: List[str] = Field(default_factory=list, description="Canonical English terms")
    tooth_groups: List[ToothGroup] = Field(default_factory=list, description="Identified tooth groups")
    intent_hints: List[str] = Field(default_factory=list, description="Intent hints")
    ambiguous: bool = Field(False, description="Whether the text is ambiguous")


class ProcessedLine(BaseModel):
    """Complete processed line with all stages."""
    raw_text: str
    n0: NormalizationResult
    t1: TokensResult
    llm_aug: Optional[LlmAugmentResult] = None