import json, os
from rdflib import Graph, RDFS, URIRef, Literal, Namespace
from dotenv import load_dotenv; load_dotenv()

OHD_OWL = os.getenv("OHD_OWL", "./data/ontology/ohd.owl")
LEXICON_PATH = os.getenv("LEXICON_PATH", "./artifacts/lexicon/ohd_lexicon.jsonl")

IAO = Namespace("http://purl.obolibrary.org/obo/IAO_")
DEF = URIRef(IAO["0000115"])  # definition
SYN = URIRef(IAO["0000118"])  # alternative term

def extract_lexicon():
    g = Graph()
    g.parse(OHD_OWL)
    out = []
    for s, _, lbl in g.triples((None, RDFS.label, None)):
        if not isinstance(lbl, Literal):
            continue
        iri = str(s)
        label = str(lbl)
        syns = [str(o) for _, _, o in g.triples((s, SYN, None)) if isinstance(o, Literal)]
        defs = [str(o) for _, _, o in g.triples((s, DEF, None)) if isinstance(o, Literal)]
        definition = defs[0] if defs else ""
        out.append({"iri": iri, "label": label, "synonyms": syns, "definition": definition})
    os.makedirs(os.path.dirname(LEXICON_PATH), exist_ok=True)
    with open(LEXICON_PATH, "w", encoding="utf-8") as f:
        for rec in out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {len(out)} rows to {LEXICON_PATH}")

if __name__ == "__main__":
    extract_lexicon()
