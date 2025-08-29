# OHD Clinic NLP (Block 0)

Setup skeleton for ontology lexicon + embeddings + Ollama wrapper.

## Quick Start

1. python -m venv .venv
2. Activate venv
3. pip install -r requirements.txt
4. Copy .env.example to .env and adjust paths.
5. Run: python src/ontology/build_lexicon.py
6. Run: python src/ontology/build_embeddings.py
7. Sanity: python src/tools/sanity_checks.py

Artifacts under artifacts/ .

## Turbo Chat Example

```python
from src.llm.ollama_client import chat_json
resp = chat_json('{"ping":1}', system='return JSON', turbo=True)
print(resp)
```
