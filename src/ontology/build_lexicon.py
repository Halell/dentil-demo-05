"""Full unique-per-entity lexicon builder.

Replaces previous naive implementation (one row per label triple) with a
single consolidated record per subject IRI including:
  - primary label (prefer English, else any)
  - synonyms: IAO:0000118 values + all other labels (excluding primary)
  - definition: first IAO:0000115 (if multiple) OR empty string

Schema per line (JSONL): {"iri","label","synonyms","definition"}
This keeps filename stable (`ohd_lexicon.jsonl`) for existing imports.
"""

import json, os
from collections import defaultdict
from rdflib import Graph, RDFS, URIRef, Literal, Namespace, RDF
from dotenv import load_dotenv; load_dotenv()

OHD_OWL = os.getenv("OHD_OWL", "./data/ontology/ohd.owl")
LEXICON_PATH = os.getenv("LEXICON_PATH", "./artifacts/lexicon/ohd_lexicon.jsonl")

IAO = Namespace("http://purl.obolibrary.org/obo/IAO_")
DEF = URIRef(IAO["0000115"])  # textual definition
SYN = URIRef(IAO["0000118"])  # alternative term / synonym

def build_full_lexicon():
    g = Graph()
    g.parse(OHD_OWL)
    labels = defaultdict(lambda: defaultdict(set))  # iri -> lang -> set(labels)
    synonyms = defaultdict(set)  # iri -> set
    definitions = defaultdict(list)  # iri -> list

    for s, p, o in g:
        if not isinstance(s, URIRef):
            continue
        if p == RDFS.label and isinstance(o, Literal):
            lang = o.language or 'und'
            labels[str(s)][lang].add(str(o))
        elif p == SYN and isinstance(o, Literal):
            synonyms[str(s)].add(str(o))
        elif p == DEF and isinstance(o, Literal):
            definitions[str(s)].append(str(o))

    records = []
    for iri, lang_map in labels.items():
        # choose primary label
        if 'en' in lang_map and lang_map['en']:
            primary = sorted(lang_map['en'])[0]
        else:
            # pick lexicographically first language then first label
            first_lang = sorted(lang_map.keys())[0]
            primary = sorted(lang_map[first_lang])[0]
        # collect secondary labels into synonyms
        extra_labels = []
        for lg, vals in lang_map.items():
            for v in vals:
                if v != primary:
                    extra_labels.append(v)
        syn_list = sorted(set(list(synonyms.get(iri, [])) + extra_labels))
        definition = definitions.get(iri, [""])[0]
        records.append({
            'iri': iri,
            'label': primary,
            'synonyms': syn_list,
            'definition': definition
        })

    os.makedirs(os.path.dirname(LEXICON_PATH), exist_ok=True)
    with open(LEXICON_PATH, 'w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(f"[build_lexicon] wrote {len(records)} entities to {LEXICON_PATH}")

if __name__ == '__main__':
    build_full_lexicon()
