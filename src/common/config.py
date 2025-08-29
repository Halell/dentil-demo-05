from pathlib import Path
import os
from dotenv import load_dotenv
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings:
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "os120b")
    EMBED_MODEL_DIR: str = os.getenv("EMBED_MODEL_DIR", "./models/embeddings/bge-m3")
    OHD_OWL: str = os.getenv("OHD_OWL", "./data/ontology/ohd.owl")
    LEXICON_PATH: str = os.getenv("LEXICON_PATH", "./artifacts/lexicon/ohd_lexicon.jsonl")
    FAISS_INDEX: str = os.getenv("FAISS_INDEX", "./artifacts/vectors/ohd.faiss")
    FAISS_META: str = os.getenv("FAISS_META", "./artifacts/vectors/ohd_meta.json")

settings = Settings()

__all__ = ["settings"]
