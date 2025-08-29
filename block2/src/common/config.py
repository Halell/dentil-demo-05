from __future__ import annotations
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Block2Config:
    CAND_ENGINE: str = os.getenv('CAND_ENGINE', 'faiss')
    TOPK_LEX: int = int(os.getenv('TOPK_LEX', '8'))
    TOPK_VEC: int = int(os.getenv('TOPK_VEC', '10'))
    TOPK_FINAL: int = int(os.getenv('TOPK_FINAL', '5'))
    W_LEX: float = float(os.getenv('W_LEX', '0.6'))
    W_VEC: float = float(os.getenv('W_VEC', '0.3'))
    W_PRIOR: float = float(os.getenv('W_PRIOR', '0.06'))
    W_CTX: float = float(os.getenv('W_CTX', '0.04'))
    LEX_FUZZY_MAX_ED: int = int(os.getenv('LEX_FUZZY_MAX_ED', '1'))
    NGRAM_MAX: int = int(os.getenv('NGRAM_MAX', '5'))
    NEO4J_URI: str = os.getenv('NEO4J_URI', 'bolt://127.0.0.1:7687')
    NEO4J_USER: str = os.getenv('NEO4J_USER', 'neo4j')
    NEO4J_PASS: str = os.getenv('NEO4J_PASS', 'pass')
    NEO4J_VECTOR_INDEX: str = os.getenv('NEO4J_VECTOR_INDEX', 'ohd_embeddings')
    NEO4J_VECTOR_DIM: int = int(os.getenv('NEO4J_VECTOR_DIM', '1024'))
    TAU_LOW: float = float(os.getenv('TAU_LOW', '0.35'))
    TAU_CONFIDENT: float = float(os.getenv('TAU_CONFIDENT', '0.65'))

CONFIG = Block2Config()
