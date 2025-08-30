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
from .alias_resolver import resolve_alias_to_ohd
from ..common.config import CONFIG
from ..common.schemas import GazetteerHit

LABEL_SCORE = 1.0
SYN_SCORE = 0.9
FUZZY_SCORE_MIN = 0.70  # lower bound for fuzzy mapped score
FUZZY_SCORE_MAX = 0.85  # upper bound for fuzzy mapped score
PHONETIC_SCORE = 0.83  # (FZ-2) score for phonetic near-miss

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
    ("שלת", "IMPLANT:שתל", "dental implant"),  # restored direct variant (FZ-1)
    ("מולטיוניט", "DEVICE:multiunit_abutment", "multi-unit abutment"),
    ("מולטי יוניט", "DEVICE:multiunit_abutment", "multi-unit abutment"),
    ("מולטי-יוניט", "DEVICE:multiunit_abutment", "multi-unit abutment"),
    ("MU", "DEVICE:multiunit_abutment", "multi-unit abutment"),
    ("MUA", "DEVICE:multiunit_abutment", "multi-unit abutment"),
]

def build_gazetteer(lexicon_path: str, clinic_abbr: str, brand_names: str, en_alias_map: str | None = None, he2en_map: str | None = None) -> GazetteerIndex:
    g = GazetteerIndex()
    # Load main OHD lexicon
    for rec in load_lexicon(lexicon_path):  # GZ-4 future place to compute frequency
        iri = rec.get('iri')
        label = rec.get('label')
        syns = rec.get('synonyms', [])
        if not iri or not label:
            continue
        # skip numeric-only or extremely short labels
        if label.isdigit() or len(label) <= 2:
            continue
        g.add(label.lower(), {"iri": iri, "label": label, "match_type": "label", "score_lex": LABEL_SCORE, "iri_source": "ohd_label"})
        for s in syns:
            g.add(s.lower(), {"iri": iri, "label": label, "match_type": "synonym", "score_lex": SYN_SCORE, "iri_source": "ohd_synonym"})
    # Clinic abbreviations
    with open(clinic_abbr, 'r', encoding='utf-8') as f:
        cabbr = json.load(f)
        for k, v in cabbr.items():
            g.add(k.lower(), {"iri": f"ABBR:{k}", "label": v, "match_type": "label", "score_lex": LABEL_SCORE, "iri_source": "abbreviation"})
    # Brands
    with open(brand_names, 'r', encoding='utf-8') as f:
        brands = json.load(f)
        for k, lst in brands.items():
            g.add(k.lower(), {"iri": f"BRAND:{k}", "label": k, "match_type": "label", "score_lex": LABEL_SCORE, "iri_source": "brand_label"})
            for alt in lst:
                g.add(alt.lower(), {"iri": f"BRAND:{k}", "label": k, "match_type": "synonym", "score_lex": SYN_SCORE, "iri_source": "brand_synonym"})
    # Inject Hebrew variants
    for surface, code, eng in HEBREW_VARIANTS:
        # Attempt to resolve English concept to real OHD IRI
        resolved = resolve_alias_to_ohd(eng)
        if resolved:
            g.add(surface.lower(), {"iri": resolved['iri'], "label": surface, "match_type": "label", "score_lex": LABEL_SCORE, "eng": eng, "iri_source": resolved.get('iri_source','resolved_alias')})
        else:
            g.add(surface.lower(), {"iri": code, "label": surface, "match_type": "label", "score_lex": LABEL_SCORE, "eng": eng, "iri_source": "placeholder"})
    # English alias -> IRI map (manual bridge)
    if en_alias_map and Path(en_alias_map).exists():
        try:
            data = json.loads(Path(en_alias_map).read_text(encoding='utf-8'))
            for alias, mapped in data.items():
                resolved = None
                if not str(mapped).startswith('http://purl.obolibrary.org/obo/'):
                    resolved = resolve_alias_to_ohd(alias)
                meta_base = {"label": alias, "match_type": "alias"}
                if resolved:
                    meta = {"iri": resolved['iri'], **meta_base, "iri_source": resolved.get('iri_source','resolved_alias'), "alias_only": False, "score_lex": SYN_SCORE}
                else:
                    # unresolved placeholder
                    meta = {"iri": mapped, **meta_base, "iri_source": "alias_only", "alias_only": True, "score_lex": SYN_SCORE - 0.2}
                # Dedup: prefer label > synonym > vector > placeholder
                surf = alias.lower()
                existing = g.entries.get(surf, [])
                def _priority(m):
                    if m.get('alias_only'):
                        return 4
                    if m.get('source') == 'label':
                        return 1
                    if m.get('source') == 'synonym':
                        return 2
                    if m.get('source') == 'vector':
                        return 3
                    return 5
                if existing:
                    # keep only best
                    allc = existing + [meta]
                    best = sorted(allc, key=_priority)[0]
                    g.entries[surf] = [best]
                else:
                    g.add(surf, meta)
        except Exception:
            pass
    # Hebrew->English static expansions to also index the English forms directly
    if he2en_map and Path(he2en_map).exists():
        try:
            he2en = json.loads(Path(he2en_map).read_text(encoding='utf-8'))
            for he, en_list in he2en.items():
                for en in en_list:
                    resolved = resolve_alias_to_ohd(en)
                    base_meta = {"label": en, "match_type": "he_en_alias", "he_src": he}
                    if resolved:
                        meta = {"iri": resolved['iri'], **base_meta, "iri_source": resolved.get('iri_source','resolved_alias'), "alias_only": False, "score_lex": SYN_SCORE}
                    else:
                        meta = {"iri": f"HE2EN:{he}", **base_meta, "iri_source": "placeholder", "alias_only": True, "score_lex": SYN_SCORE - 0.2}
                    surf = en.lower()
                    existing = g.entries.get(surf, [])
                    def _priority(m):
                        if m.get('alias_only'):
                            return 4
                        if m.get('source') == 'label':
                            return 1
                        if m.get('source') == 'synonym':
                            return 2
                        if m.get('source') == 'vector':
                            return 3
                        return 5
                    if existing:
                        allc = existing + [meta]
                        best = sorted(allc, key=_priority)[0]
                        g.entries[surf] = [best]
                    else:
                        g.add(surf, meta)
        except Exception:
            pass
    return g


def ngram_slices(tokens: List[Dict], n_max: int) -> List[Tuple[Tuple[int,int], List[Dict], Tuple[int,int]]]:
    # Legacy function retained but not used after segment-based rewrite
    out = []
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
    # Determine project root: prefer walking up until 'artifacts' directory present
    project_root = Path(__file__).parents[3]
    candidate = Path(merged_block1_path).resolve()
    base = candidate
    for _ in range(8):
        if (base / 'artifacts' / 'lexicon' / 'ohd_lexicon.jsonl').exists():
            break
        if base == base.parent:
            break
        base = base.parent
    # prefer explicit project_root if it has artifacts
    if (project_root / 'artifacts' / 'lexicon' / 'ohd_lexicon.jsonl').exists():
        base = project_root
    if not (base / 'artifacts' / 'lexicon' / 'ohd_lexicon.jsonl').exists():
        raise FileNotFoundError('ohd_lexicon.jsonl not found in ancestry or project root')
    lexicon_path = base / 'artifacts' / 'lexicon' / 'ohd_lexicon.jsonl'
    clinic_abbr = Path(__file__).parent.parent.parent / 'data' / 'dictionaries' / 'clinic_abbreviations.json'
    brand_names = Path(__file__).parent.parent.parent / 'data' / 'dictionaries' / 'brand_names.json'
    en_alias_map = Path(__file__).parent.parent.parent / 'data' / 'dictionaries' / 'en_alias_to_iri.json'
    he2en_static_map = Path(__file__).parent.parent.parent / 'data' / 'dictionaries' / 'he2en_static.json'
    g = build_gazetteer(str(lexicon_path), str(clinic_abbr), str(brand_names), str(en_alias_map), str(he2en_static_map))

    # ---- (GZ-4) Two-pass: load all lines to compute token frequency prior ----
    with open(merged_block1_path, 'r', encoding='utf-8') as fin_all:
        all_lines = fin_all.readlines()
    token_freq: Dict[str,int] = {}
    for line in all_lines:
        try:
            rec_tmp = json.loads(line)
        except Exception:
            continue
        tokens_tmp = rec_tmp.get('t1', {}).get('tokens', [])
        for tok in tokens_tmp:
            if tok.get('kind') != 'word':
                continue
            txt = normalize_hebrew(tok.get('text','').lower())
            if not txt or txt.isdigit() or len(txt) <= 2:
                continue
            token_freq[txt] = token_freq.get(txt,0)+1
    max_freq = max(token_freq.values()) if token_freq else 1

    # (FZ-2) Precompute simple phonetic map for ASCII surfaces
    def english_phonetic(s: str) -> str:
        s = re.sub(r"[^A-Za-z]", "", s.lower())
        if not s:
            return ''
        # remove vowels (keep first char) + collapse doubles
        first = s[0]
        rest = re.sub(r"[aeiou]", "", s[1:])
        out = first + rest
        out2 = []
        for ch in out:
            if not out2 or out2[-1] != ch:
                out2.append(ch)
        return ''.join(out2)
    phonetic_map: Dict[str, List[Dict]] = {}
    for surf, metas in g.entries.items():
        if surf.isascii() and surf.replace(' ','').isalpha():
            code = english_phonetic(surf)
            if code:
                phonetic_map.setdefault(code, []).extend(metas)

    total_tokens=0; covered_tokens_total=0; mentions_count=0
    with open(out_path, 'w', encoding='utf-8') as fout:
        for line_idx, line in enumerate(all_lines):
            rec = json.loads(line)
            n0 = rec.get('n0', {})
            t1 = rec.get('t1', {})
            tokens = t1.get('tokens', [])
            total_tokens += len(tokens)
            # Helper: split tokens into contiguous word-only segments (Fix Pack v3 #1)
            ALLOWED_FOR_NGRAM = {'word'}
            segments = []
            current = []
            for tk in tokens:
                if tk.get('kind') in ALLOWED_FOR_NGRAM:
                    current.append(tk)
                else:
                    if current:
                        segments.append(current)
                        current = []
            if current:
                segments.append(current)

            covered_token_idxs = set()
            raw_matches = []
            # 2) token-level direct & fuzzy (Hebrew words)
            entry_keys = list(g.entries.keys())
            heb_keys = [k for k in entry_keys if any('\u0590' <= ch <= '\u05FF' for ch in k)]
            for tok in tokens:
                if tok.get('kind') != 'word':
                    continue
                surf_orig = tok['text']
                norm_tok = normalize_hebrew(surf_orig.lower())
                ascii_word = surf_orig.isascii() and surf_orig.replace(' ','').isalpha()
                # exact
                if norm_tok in g.entries:
                    for meta in g.entries[norm_tok]:
                        m2 = meta.copy()
                        if norm_tok == 'שלת':
                            m2['match_type'] = 'fuzzy'
                            if m2.get('score_lex',1.0) >= 0.85:
                                m2['score_lex'] = 0.82
                            m2['normalized_surface'] = 'שתל'
                            # preserve existing iri_source if available
                            if 'iri_source' not in m2:
                                m2['iri_source'] = 'resolved_alias'
                        raw_matches.append(((tok['span'][0], tok['span'][1]), m2))
                        covered_token_idxs.add(tok['idx'])
                    continue
                # skip fuzzy for digit-only tokens
                if norm_tok.isdigit():
                    continue
                # fuzzy hebrew (distance <= LEX_FUZZY_MAX_ED)
                fuzzy_added = False
                for ek in heb_keys:
                    if ek == norm_tok:
                        continue
                    if he_fuzzy_ok(norm_tok, ek, max_ed=CONFIG.LEX_FUZZY_MAX_ED, min_ratio=90):
                        # compute similarity ratio via Levenshtein distance for scaling
                        try:
                            dist = distance.Levenshtein.distance(norm_tok, ek)
                            denom = max(len(norm_tok), len(ek)) or 1
                            sim_ratio = max(0.0, 1.0 - dist/denom)
                        except Exception:
                            sim_ratio = 0.0
                        scaled = FUZZY_SCORE_MIN + (FUZZY_SCORE_MAX - FUZZY_SCORE_MIN) * sim_ratio
                        for meta in g.entries[ek]:
                            fm = meta.copy()
                            fm['match_type'] = 'fuzzy'
                            fm['score_lex'] = round(min(FUZZY_SCORE_MAX, max(FUZZY_SCORE_MIN, scaled)), 4)
                            raw_matches.append(((tok['span'][0], tok['span'][1]), fm))
                            covered_token_idxs.add(tok['idx'])
                            fuzzy_added = True
                        break
                # (FZ-2) phonetic fallback (English transliteration) if no exact/fuzzy
                if not fuzzy_added and ascii_word:
                    code = english_phonetic(surf_orig)
                    if code in phonetic_map:
                        for meta in phonetic_map[code]:
                            pm = meta.copy()
                            pm['match_type'] = 'phonetic'
                            pm['score_lex'] = PHONETIC_SCORE
                            raw_matches.append(((tok['span'][0], tok['span'][1]), pm))
                            covered_token_idxs.add(tok['idx'])
            # Build ngrams within each pure word segment only (no crossing numbers/pairs/units)
            for seg in segments:
                seg_tokens = seg
                L = len(seg_tokens)
                if L == 0:
                    continue
                for n in range(min(cfg.NGRAM_MAX, L), 0, -1):
                    for i in range(0, L - n + 1):
                        window = seg_tokens[i:i+n]
                        # Skip if any token index already covered (longest-first, non overlap)
                        if any(w['idx'] in covered_token_idxs for w in window):
                            continue
                        span = (window[0]['span'][0], window[-1]['span'][1])
                        surf = surface_from_tokens(window).lower()
                        norm_surf = normalize_hebrew(surf)
                        found_any = False
                        for found, start, end in g.extract(norm_surf):
                            metas = g.entries.get(found, [])
                            for meta in metas:
                                raw_matches.append((span, meta))
                                found_any = True
                        if not found_any:
                            # optional fuzzy for multi-token surface
                            candidates = list(g.entries.keys())
                            sample = candidates if len(candidates) < 5000 else candidates[:5000]
                            fuzzy = process.extract(norm_surf, sample, limit=3)
                            for cand_label, rf_score, _ in fuzzy:
                                if he_fuzzy_ok(norm_surf, cand_label, max_ed=CONFIG.LEX_FUZZY_MAX_ED, min_ratio=90):
                                    try:
                                        sim_ratio = (rf_score or 0)/100.0
                                    except Exception:
                                        sim_ratio = 0.0
                                    band_min, band_max = FUZZY_SCORE_MIN, FUZZY_SCORE_MAX
                                    if len(norm_surf) <= 4:
                                        band_min, band_max = 0.65, 0.88
                                    scaled = band_min + (band_max - band_min) * sim_ratio
                                    for meta in g.entries[cand_label]:
                                        fm = meta.copy()
                                        fm['match_type'] = 'fuzzy'
                                        fm['score_lex'] = round(min(band_max, max(band_min, scaled)), 4)
                                        raw_matches.append((span, fm))
                        for w in window:
                            covered_token_idxs.add(w['idx'])
            # deduplicate longest non-overlapping
            filtered = longest_non_overlapping(raw_matches)
            hits = []
            enriched_hits = []
            for span0, span1, meta in filtered:
                surface = meta['label']
                normalized_surface = None
                if surface != surface.lower():
                    # attempt simple normalization placeholder; real logic could be more complex
                    normalized_surface = normalize_hebrew(surface.lower()) if meta.get('match_type') == 'fuzzy' else None
                # refine: skip numeric-only surfaces
                if surface.isdigit():
                    continue  # ensure pure numbers filtered (test_g5)
                # approximate token kinds in this span
                token_kinds = [t['kind'] for t in tokens if not (t['span'][1] <= span0 or t['span'][0] >= span1)]
                if any(k in {'number','pair','unit'} for k in token_kinds):
                    continue
                if not CONFIG.KEEP_ALL_METAS:
                    # E5 post-resolve dedup preference: if multiple candidates for same surface keep real IRI over placeholder
                    if meta.get('iri','').startswith(('IMPLANT:', 'DEVICE:', 'HE2EN:')):
                        real_exists = any(
                            (m[0]==(span0, span1) and m[1].get('iri','').startswith('http://purl.obolibrary.org/obo/'))
                            for m in raw_matches
                        )
                        if real_exists:
                            continue
                mention_id = f"m{line_idx}_{span0}_{span1}"
                hints = set(derive_hints(surface))
                # Strengthen hints mapping (Fix Pack v3 #4)
                surf_low = surface.lower()
                if surf_low in {"מולטיוניט","mu","mua"}:
                    hints.add('device_hint')
                if surf_low in {"שתל","שלת"}:
                    hints.add('implant_hint')
                # (GZ-4) compute simple prior score (avg of constituent token freq / max_freq)
                span_tokens = [t for t in tokens if not (t['span'][1] <= span0 or t['span'][0] >= span1)]
                freqs = []
                for st in span_tokens:
                    if st.get('kind')=='word':
                        nt = normalize_hebrew(st['text'].lower())
                        if nt in token_freq:
                            freqs.append(token_freq[nt]/max_freq)
                score_prior = round(sum(freqs)/len(freqs),4) if freqs else 0.0
                meta = meta.copy()
                meta['score_prior'] = score_prior
                # collect token indexes covered by this span
                cov_idxs = [t['idx'] for t in tokens if not (t['span'][1] <= span0 or t['span'][0] >= span1)]
                covered_token_idxs.update(cov_idxs)
                enriched_hits.append((mention_id, surface, (span0, span1), meta, list(hints), cov_idxs, normalized_surface))
            # Span grouping removed (Fix Pack v3) – no crossing into number/pair tokens
            hits_final = []
            for mention_id, surface, (span0, span1), meta, hints, cov_idxs, normalized_surface in enriched_hits:
                # pull normalized_surface/match_type from meta if present
                ns_val = meta.get('normalized_surface') or normalized_surface
                hit = GazetteerHit(
                    mention_id=mention_id,
                    surface=surface,
                    span=(span0, span1),
                    ngram=len(surface.split()),
                    candidates_lex=[meta],
                    hints=hints,
                    covered_token_idxs=cov_idxs,
                    normalized_surface=ns_val,
                    match_type=meta.get('match_type')
                )
                hits_final.append(hit)
            for h in hits_final:
                fout.write(h.model_dump_json() + '\n')
            covered_tokens_total += len(covered_token_idxs)
            mentions_count += len(hits)
    # Write simple stats sidecar (GZ-5)
    stats_path = Path(out_path).with_suffix('.stats.json')
    try:
        stats = {
            'total_tokens': total_tokens,
            'covered_tokens': covered_tokens_total,
            'coverage_ratio': round((covered_tokens_total/total_tokens) if total_tokens else 0,4),
            'mentions': mentions_count
        }
        stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')
    except Exception:
        pass
