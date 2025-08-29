from pydantic import BaseModel
from typing import List, Optional

class LexiconEntry(BaseModel):
    iri: str
    label: str
    synonyms: List[str] = []
    definition: Optional[str] = ""

__all__ = ["LexiconEntry"]
