from __future__ import annotations
import json
from typing import List, Dict
from ..common.config import CONFIG
from ..common.schemas import VectorHit

def vector_candidates_neo4j(line: Dict, topk: int) -> List[VectorHit]:
    # Placeholder simple passthrough until vector embedding retrieval implemented in Neo4j
    return []

def run_vector_neo4j(merged_block1_path: str, out_path: str) -> None:
    # Currently stub; writes nothing if engine not ready
    with open(out_path, 'w', encoding='utf-8') as fout:
        pass
