from __future__ import annotations
import json, uuid, re
from pathlib import Path
from typing import List, Dict, Tuple
try:
    from flashtext import KeywordProcessor  # type: ignore
except Exception:  # fallback minimal shim
    class KeywordProcessor:  # type: ignore
        def __init__(self, case_sensitive=False):
            self._keys = {}
        def add_keyword(self, k, v):
            self._keys[k] = v
        def extract_keywords(self, text, span_info=False):
            out = []
            for k,v in self._keys.items():
                idx = text.find(k)
                if idx != -1:
                    if span_info:
                        out.append((v, idx, idx+len(k)))
                    else:
                        out.append(v)
            return out
    
try:
    from rapidfuzz import process, distance  # type: ignore
except Exception:
    class _Distance:
        class Levenshtein:
            @staticmethod
            def distance(a,b):
                return abs(len(a)-len(b))
    distance = _Distance()
    class _Process:
        @staticmethod
        def extract(q, choices, limit=3):
            return [(c, 0, i) for i,c in enumerate(choices[:limit])]
    process = _Process()
from .utils import normalize_hebrew
from ._he_norm import he_fuzzy_ok
from ..common.config import CONFIG
from ..common.schemas import GazetteerHit

LABEL_SCORE = 1.0
SYN_SCORE = 0.9
FUZZY_SCORE = 0.7

class GazetteerIndex:
    def __init__(self):
        self.kp = KeywordProcessor(case_sensitive=False)
        self.entries: Dict[str, List[Dict]] = {}

    def add(self, surface: str, meta: Dict):
        surf = normalize_hebrew(surface)
        if surf not in self.entries:
            self.entries[surf] = []
            self.kp.add_keyword(surf, surf)  # store canonical surface as value
        self.entries[surf].append(meta)

    def extract(self, text: str) -> List[Tuple[str, int, int]]:
        return self.kp.extract_keywords(text, span_info=True)


def load_lexicon(lexicon_path: str):
    with open(lexicon_path, 'r', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)


HEBREW_VARIANTS = [
    ("שתל", "IMPLANT:שתל", "dental implant"),
    ("שלת", "IMPLANT:שתל", "dental implant"),  # common typo
    ("מולטיוניט", "DEVICE:multiunit_abutment", "multi-unit abutment"),
    ("מולטי יוניט", "DEVICE:multiunit_abutment", "multi-unit abutment"),
    ("מולטי-יוניט", "DEVICE:multiunit_abutment", "multi-unit abutment"),
    ("MU", "DEVICE:multiunit_abutment", "multi-unit abutment"),
    ("MUA", "DEVICE:multiunit_abutment", "multi-unit abutment"),
]

def build_gazetteer(lexicon_path: str, clinic_abbr: str, brand_names: str, en_alias_map: str | None = None, he2en_map: str | None = None) -> GazetteerIndex:
    g = GazetteerIndex()
    # Load main OHD lexicon
    for rec in load_lexicon(lexicon_path):
        iri = rec.get('iri')
        label = rec.get('label')
        syns = rec.get('synonyms', [])
        if not iri or not label:
            continue
        # skip numeric-only or extremely short labels
        if label.isdigit() or len(label) <= 2:
            continue
        g.add(label.lower(), {"iri": iri, "label": label, "match_type": "label", "score_lex": LABEL_SCORE})
        for s in syns:
            g.add(s.lower(), {"iri": iri, "label": label, "match_type": "synonym", "score_lex": SYN_SCORE})
    # Clinic abbreviations
    with open(clinic_abbr, 'r', encoding='utf-8') as f:
        cabbr = json.load(f)
        for k, v in cabbr.items():
            g.add(k.lower(), {"iri": f"ABBR:{k}", "label": v, "match_type": "label", "score_lex": LABEL_SCORE})
    # Brands
    with open(brand_names, 'r', encoding='utf-8') as f:
        brands = json.load(f)
        for k, lst in brands.items():
            g.add(k.lower(), {"iri": f"BRAND:{k}", "label": k, "match_type": "label", "score_lex": LABEL_SCORE})
            for alt in lst:
                g.add(alt.lower(), {"iri": f"BRAND:{k}", "label": k, "match_type": "synonym", "score_lex": SYN_SCORE})
    # Inject Hebrew variants
    for surface, code, eng in HEBREW_VARIANTS:
        g.add(surface.lower(), {"iri": code, "label": surface, "match_type": "label", "score_lex": LABEL_SCORE, "eng": eng})
    # English alias -> IRI map (manual bridge)
    if en_alias_map and Path(en_alias_map).exists():
        try:
            data = json.loads(Path(en_alias_map).read_text(encoding='utf-8'))
            for alias, iri in data.items():
                g.add(alias.lower(), {"iri": iri, "label": alias, "match_type": "alias", "score_lex": SYN_SCORE})
        except Exception:
            pass
    # Hebrew->English static expansions to also index the English forms directly
    if he2en_map and Path(he2en_map).exists():
        try:
            he2en = json.loads(Path(he2en_map).read_text(encoding='utf-8'))
            for he, en_list in he2en.items():
                for en in en_list:
                    g.add(en.lower(), {"iri": f"HE2EN:{he}", "label": en, "match_type": "he_en_alias", "score_lex": SYN_SCORE})
        except Exception:
            pass
    return g


def ngram_slices(tokens: List[Dict], n_max: int) -> List[Tuple[Tuple[int,int], List[Dict], Tuple[int,int]]]:
    out = []
    for i in range(len(tokens)):
        for n in range(n_max, 0, -1):
            seg = tokens[i:i+n]
            if len(seg) < n:
                continue
            span = (seg[0]['span'][0], seg[-1]['span'][1])
            out.append((span, seg, (seg[0]['idx'], seg[-1]['idx'])))
    out.sort(key=lambda x: (x[0][1]-x[0][0]), reverse=True)
    return out


def longest_non_overlapping(matches: List[Tuple[Tuple[int,int], Dict]]):
    taken = []
    for span, meta in matches:
        if any(not (span[1] <= s0 or span[0] >= s1) for s0, s1, _ in taken):
            continue
        taken.append((span[0], span[1], meta))
    return taken


def surface_from_tokens(tokens: List[Dict]):
    return ' '.join(t['text'] for t in tokens)


HINT_KEYWORDS = {
    'device_hint': ['abutment', 'implant', 'שתל'],
    'material_hint': ['zirconia', 'titanium'],
    'finding_hint': ['caries']
}


def derive_hints(surface: str) -> List[str]:
    s = surface.lower()
    out = []
    for hint, words in HINT_KEYWORDS.items():
        if any(w in s for w in words):
            out.append(hint)
    return out


def run_gazetteer(merged_block1_path: str, out_path: str) -> None:
    cfg = CONFIG
    base = Path(merged_block1_path).parents[2]  # adjust if needed
    lexicon_path = base / 'artifacts' / 'lexicon' / 'ohd_lexicon.jsonl'
    clinic_abbr = Path(__file__).parent.parent.parent / 'data' / 'dictionaries' / 'clinic_abbreviations.json'
    brand_names = Path(__file__).parent.parent.parent / 'data' / 'dictionaries' / 'brand_names.json'
    en_alias_map = Path(__file__).parent.parent.parent / 'data' / 'dictionaries' / 'en_alias_to_iri.json'
    he2en_static_map = Path(__file__).parent.parent.parent / 'data' / 'dictionaries' / 'he2en_static.json'
    g = build_gazetteer(str(lexicon_path), str(clinic_abbr), str(brand_names), str(en_alias_map), str(he2en_static_map))

    with open(merged_block1_path, 'r', encoding='utf-8') as fin, open(out_path, 'w', encoding='utf-8') as fout:
        for line_idx, line in enumerate(fin):
            rec = json.loads(line)
            n0 = rec.get('n0', {})
            t1 = rec.get('t1', {})
            tokens = t1.get('tokens', [])
            # cover mask for token idxs already matched by fast-path
            covered_token_idxs = set()
            # Build ngram candidates (defer until after token fast path)
            raw_matches = []
            # 1) token-level direct & fuzzy (Hebrew words)
            entry_keys = list(g.entries.keys())
            heb_keys = [k for k in entry_keys if any('\u0590' <= ch <= '\u05FF' for ch in k)]
            for tok in tokens:
                if tok.get('kind') != 'word':
                    continue
                surf_orig = tok['text']
                norm_tok = normalize_hebrew(surf_orig.lower())
                # exact
                if norm_tok in g.entries:
                    for meta in g.entries[norm_tok]:
                        raw_matches.append(((tok['span'][0], tok['span'][1]), meta))
                        covered_token_idxs.add(tok['idx'])
                    continue
                # fuzzy hebrew (distance <= LEX_FUZZY_MAX_ED)
                fuzzy_added = False
                for ek in heb_keys:
                    if ek == norm_tok:
                        continue
                    if he_fuzzy_ok(norm_tok, ek, max_ed=CONFIG.LEX_FUZZY_MAX_ED, min_ratio=90):
                        for meta in g.entries[ek]:
                            fm = meta.copy()
                            fm['match_type'] = 'fuzzy'
                            fm['score_lex'] = FUZZY_SCORE
                            raw_matches.append(((tok['span'][0], tok['span'][1]), fm))
                            covered_token_idxs.add(tok['idx'])
                            fuzzy_added = True
                        break
            # Now build ngrams only over uncovered word tokens
            slices = [s for s in ngram_slices(tokens, cfg.NGRAM_MAX) if not any(i in covered_token_idxs for i in range(s[2][0], s[2][1]+1))]
            for span, seg, tok_idx_rng in slices:
                surf = surface_from_tokens(seg).lower()
                norm_surf = normalize_hebrew(surf)
                for found, start, end in g.extract(norm_surf):
                    metas = g.entries.get(found, [])
                    for meta in metas:
                        raw_matches.append((span, meta))
                # Fuzzy fallback if no exact meta added for this ngram
                if not any(m[0] == span for m in raw_matches):
                    # approximate match using Levenshtein edit distance threshold
                    candidates = list(g.entries.keys())
                    # limit search size heuristically
                    sample = candidates if len(candidates) < 5000 else candidates[:5000]
                    # Use rapidfuzz process to get top few similar strings
                    fuzzy = process.extract(norm_surf, sample, limit=3)
                    for cand_label, score, _ in fuzzy:
                        if he_fuzzy_ok(norm_surf, cand_label, max_ed=CONFIG.LEX_FUZZY_MAX_ED, min_ratio=90):
                            for meta in g.entries[cand_label]:
                                fm = meta.copy()
                                fm['match_type'] = 'fuzzy'
                                fm['score_lex'] = FUZZY_SCORE
                                raw_matches.append((span, fm))
            # deduplicate longest non-overlapping
            filtered = longest_non_overlapping(raw_matches)
            hits = []
            for span0, span1, meta in filtered:
                surface = meta['label']
                # refine: skip numeric-only surfaces
                if surface.isdigit():
                    continue
                # approximate token kinds in this span
                token_kinds = [t['kind'] for t in tokens if not (t['span'][1] <= span0 or t['span'][0] >= span1)]
                if any(k in {'number','pair','unit'} for k in token_kinds):
                    continue
                mention_id = f"m{line_idx}_{span0}_{span1}"
                hints = derive_hints(surface)
                hit = GazetteerHit(
                    mention_id=mention_id,
                    surface=surface,
                    span=(span0, span1),
                    ngram=len(surface.split()),
                    candidates_lex=[meta],
                    hints=hints,
                    covered_token_idxs=None
                )
                hits.append(hit)
            for h in hits:
                fout.write(h.model_dump_json() + '\n')
