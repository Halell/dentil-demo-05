from __future__ import annotations
from pydantic import BaseModel
from typing import List, Tuple, Dict, Any

class GazetteerHit(BaseModel):
    mention_id: str
    surface: str
    span: Tuple[int, int]
    ngram: int
    candidates_lex: List[Dict]
    hints: List[str] = []
    covered_token_idxs: Tuple[int,int] | None = None

class VectorHit(BaseModel):
    mention_id: str
    surface: str
    context: str
    span: Tuple[int, int] | None = None
    candidates_vec: List[Dict]

class MentionCandidates(BaseModel):
    mention_id: str
    surface: str
    span: Tuple[int, int]
    hints: List[str] = []
    candidates: List[Dict]
    confident_singleton: bool = False

class Block2LineOutput(BaseModel):
    line_id: str
    gazetteer: List[GazetteerHit]
    vector: List[VectorHit]
    merged: List[MentionCandidates]
