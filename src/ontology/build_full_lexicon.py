from __future__ import annotations
"""Full ontology extraction: nodes (entities) + edges (relationships) + annotations.

Goal: Capture all classes, properties, and their relationships from OHD OWL so downstream
components can enrich lexical + structural reasoning.

Outputs (JSONL):
  artifacts/lexicon/ohd_nodes_full.jsonl  -> one JSON object per entity
  artifacts/lexicon/ohd_edges_full.jsonl  -> one JSON object per edge (subject, predicate, object)

Node schema:
  {
    "iri": str,
    "types": [str],             # rdf:type IRIs
    "label": str | null,        # first rdfs:label (lang preference en, else any)
    "labels": { lang: [vals] }, # all labels by lang
    "synonyms": [str],          # IAO:0000118 values
    "definitions": [str],       # IAO:0000115 values
    "annotations": { pred_iri: [literal strings] }  # all literal annotations (excluding label/definition/synonym already promoted)
  }

Edge schema:
  {
    "subj": str,
    "pred": str,
    "obj": str
  }

"100%" coverage caveat: We include every triple where object is a URIRef (non-literal) except
annotation predicates (label, synonym, definition) and rdf:type (captured in node.types).
All literal annotation values are retained under node.annotations.
"""

import os, json, sys
from collections import defaultdict
from rdflib import Graph, URIRef, Literal, RDF, RDFS, OWL
from rdflib.namespace import Namespace
from dotenv import load_dotenv

load_dotenv()

OHD_OWL = os.getenv("OHD_OWL", "./data/ontology/ohd.owl")
OUT_DIR = os.getenv("LEXICON_DIR", "./artifacts/lexicon")
NODES_PATH = os.path.join(OUT_DIR, "ohd_nodes_full.jsonl")
EDGES_PATH = os.path.join(OUT_DIR, "ohd_edges_full.jsonl")

IAO = Namespace("http://purl.obolibrary.org/obo/IAO_")
ALT_TERM = URIRef(IAO["0000118"])  # alternative term / synonym
DEFINITION = URIRef(IAO["0000115"])  # textual definition

# Annotation predicates we elevate or skip as edges
ANNOT_LABEL = RDFS.label
ANNOT_SYNONYM = ALT_TERM
ANNOT_DEF = DEFINITION
CORE_ANNOT_SKIP = {ANNOT_LABEL, ANNOT_SYNONYM, ANNOT_DEF}

def load_graph(path: str) -> Graph:
    g = Graph()
    g.parse(path)
    return g

def build_indices(g: Graph):
    nodes: dict[str, dict] = {}
    labels_by_lang: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    synonyms: dict[str, set[str]] = defaultdict(set)
    definitions: dict[str, set[str]] = defaultdict(set)
    types: dict[str, set[str]] = defaultdict(set)
    annotations: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    edges: list[tuple[str, str, str]] = []

    # First pass: collect everything
    for s, p, o in g:
        if isinstance(s, URIRef):
            s_iri = str(s)
        else:
            continue  # we care about URI subjects only
        if p == RDF.type and isinstance(o, URIRef):
            types[s_iri].add(str(o))
            continue
        if p == RDFS.label and isinstance(o, Literal):
            lang = o.language or "und"
            labels_by_lang[s_iri][lang].add(str(o))
            continue
        if p == ALT_TERM and isinstance(o, Literal):
            synonyms[s_iri].add(str(o))
            continue
        if p == DEFINITION and isinstance(o, Literal):
            definitions[s_iri].add(str(o))
            continue
        # literal annotations (other predicates)
        if isinstance(o, Literal):
            annotations[s_iri][str(p)].add(str(o))
            continue
        # object edge
        if isinstance(o, URIRef):
            # skip annotation predicates already handled, and rdf:type
            if p in CORE_ANNOT_SKIP or p == RDF.type:
                continue
            edges.append((s_iri, str(p), str(o)))

    # Assemble node records
    for iri in set(list(labels_by_lang.keys()) + list(synonyms.keys()) + list(definitions.keys()) + list(types.keys()) + list(annotations.keys())):
        lang_map = labels_by_lang.get(iri, {})
        # choose primary label: prefer English 'en', else any deterministic
        primary_label = None
        if 'en' in lang_map and lang_map['en']:
            primary_label = sorted(lang_map['en'])[0]
        elif lang_map:
            # pick smallest lang code then first label
            first_lang = sorted(lang_map.keys())[0]
            primary_label = sorted(lang_map[first_lang])[0]
        rec = {
            'iri': iri,
            'types': sorted(types.get(iri, [])),
            'label': primary_label,
            'labels': {lg: sorted(vals) for lg, vals in lang_map.items()},
            'synonyms': sorted(synonyms.get(iri, [])),
            'definitions': sorted(definitions.get(iri, [])),
            'annotations': {pred: sorted(vals) for pred, vals in annotations.get(iri, {}).items()},
        }
        nodes[iri] = rec

    return nodes, edges

def write_jsonl(path: str, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

def main():
    print(f"[full_lexicon] parsing graph: {OHD_OWL}")
    g = load_graph(OHD_OWL)
    print(f"[full_lexicon] triples: {len(g)}")
    nodes, edges = build_indices(g)
    print(f"[full_lexicon] nodes: {len(nodes)}  edges: {len(edges)}")
    write_jsonl(NODES_PATH, nodes.values())
    edge_objs = [{"subj": s, "pred": p, "obj": o} for s, p, o in edges]
    write_jsonl(EDGES_PATH, edge_objs)
    print(f"[full_lexicon] wrote nodes -> {NODES_PATH}")
    print(f"[full_lexicon] wrote edges -> {EDGES_PATH}")

if __name__ == "__main__":
    sys.exit(main())
