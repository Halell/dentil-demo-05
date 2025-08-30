import os, json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv; load_dotenv()

LEXICON_PATH = os.getenv("LEXICON_PATH", "./artifacts/lexicon/ohd_lexicon.jsonl")
EMBED_MODEL_DIR = os.getenv("EMBED_MODEL_DIR", "./models/embeddings/bge-m3")
FAISS_INDEX = os.getenv("FAISS_INDEX", "./artifacts/vectors/ohd.faiss")
FAISS_META = os.getenv("FAISS_META", "./artifacts/vectors/ohd_meta.json")

def load_lexicon():
    entries, texts = [], []
    with open(LEXICON_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            entries.append(rec)
            txt = rec["label"]
            if rec.get("synonyms"):
                txt += " || " + " | ".join(rec["synonyms"])
            if rec.get("definition"):
                txt += " ## " + rec["definition"]
            texts.append(txt)
    return entries, texts

def build_index():
    entries, texts = load_lexicon()
    # Resolve model path with fallbacks
    candidates = []
    if os.path.isdir(EMBED_MODEL_DIR):
        candidates.append(EMBED_MODEL_DIR)
    else:
        candidates.append(EMBED_MODEL_DIR)  # treat as HF repo id
    # Common public fallbacks
    candidates.extend([
        "sentence-transformers/all-MiniLM-L6-v2",
        "all-MiniLM-L6-v2"
    ])
    last_err = None
    model = None
    loaded_name = None
    for cand in candidates:
        try:
            print(f"Trying embedding model: {cand}")
            model = SentenceTransformer(cand)
            print(f"Loaded model: {cand}")
            loaded_name = cand
            break
        except Exception as e:  # noqa
            last_err = e
            print(f"Failed loading {cand}: {e}")
    if model is None:
        raise RuntimeError(f"Could not load any embedding model. Last error: {last_err}")
    embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)
    d = embs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embs)
    os.makedirs(os.path.dirname(FAISS_INDEX), exist_ok=True)
    faiss.write_index(index, FAISS_INDEX)
    # Backward + forward compatible metadata: include simple rows for fast lookup & full entries
    rows = [{"iri": r.get("iri"), "label": r.get("label")} for r in entries]
    meta_obj = {
        "dim": int(d),
        "count": len(entries),
        "model_name": loaded_name or candidates[0],
        "rows": rows,
        "entries": entries,
    }
    with open(FAISS_META, "w", encoding="utf-8") as f:
        json.dump(meta_obj, f, ensure_ascii=False)
    print(f"FAISS index built: {FAISS_INDEX}, dim={d}, n={len(entries)}")

if __name__ == "__main__":
    build_index()
