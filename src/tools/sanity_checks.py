import os, json, faiss, httpx
from dotenv import load_dotenv; load_dotenv()

def check_ollama():
    host = os.getenv("OLLAMA_HOST","http://127.0.0.1:11434")
    r = httpx.get(host)
    print("Ollama OK:", r.status_code, host)

def check_faiss():
    fa = os.getenv("FAISS_INDEX"); meta = os.getenv("FAISS_META")
    idx = faiss.read_index(fa); print("FAISS OK: n_total =", idx.ntotal)
    with open(meta,"r",encoding="utf-8") as f: m = json.load(f)
    print("META OK:", m.get("count"), "entries")

if __name__ == "__main__":
    check_ollama()
    check_faiss()
