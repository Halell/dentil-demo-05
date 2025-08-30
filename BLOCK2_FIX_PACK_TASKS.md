# Block 2 Fix Pack – Task List

מקור: מסמך "תיקונים". מטרת החבילה: מעבר מפלייסהולדרים ל‑OHD IRIs אמיתיים + שיפור סקורינג וחילוץ.

---
## A) Alias → OHD אמיתי
- [ ] A1 טען `ohd_lexicon.jsonl` למבנה חיפוש (label+synonyms→iri)
- [ ] A2 פונקציה `resolve_alias_to_ohd(alias_text, lexicon_index, faiss, embedder)`
  - exact (case‑insensitive EN) על label/synonyms
  - contains/normalized match
  - vector Top‑3 (τ ≥ 0.6)
- [ ] A3 אינטגרציה ב`t2_gazetteer`: אם alias map נותן ערך לא URL → נסה resolve; הצלחה ⇒ החלף ל IRI מלא וסמן `iri_source=resolved_alias`
- [ ] A4 כישלון resolve ⇒ `iri_source=alias_only` + penalty בהייבריד

## B) שכבת וקטור – שאילתות EN והרחבה
- [ ] B1 פונקציית `build_query_text`: canonical_terms > he2en_static[surface] > surface
- [ ] B2 הוסף עד 2 שכני טוקן (לפני/אחרי) לסוף המחרוזת
- [ ] B3 אימות מודל: בדוק ש`meta.model_name` == המודל הטעון; אחרת לוג אזהרה
- [ ] B4 לוג DEBUG: query_text + top3 IRIs (scores raw)

## C) Scoring / Hybrid
- [ ] C1 בטל כל נרמול ללקסיקון (השאר 1.0 / 0.9 / fuzzy_band)
- [ ] C2 נרמול רק לוקטורים -> `norm_vec`
- [ ] C3 Re‑weight דינמי: משקלל רק רכיבים נוכחים (W_LEX,W_VEC,...) / ΣW
- [ ] C4 `confident_singleton=True` אם מועמד יחיד עם `score_lex≥0.9` ו`iri_source∈{ohd_label,ohd_synonym,resolved_alias}`
- [ ] C5 Penalty ל`alias_only` (‑0.05 עד ‑0.1 לציון הסופי אחרי חיבור)

## D) הרחבת מבנה מועמד
- [ ] D1 הוסף שדות:
```json
{"iri":"...","label":"...",
 "scores":{"lex":..,"vec":..,"prior":null,"ctx":null},
 "score_final":...,"iri_source":"ohd_label|ohd_synonym|resolved_alias|alias_only",
 "normalized_surface":"..."}
```
- [ ] D2 העברת מקור פאזי ל`normalized_surface` (מקור surface נשאר מקורי)
- [ ] D3 שמירת מקור מקור (label/synonym/fuzzy)

## E) Hints + Normalization
- [ ] E1 עדכן מפה: `implant_hint` (שתל/שלת/implant) ; `device_hint` (מולטיוניט/MU/abutment)
- [ ] E2 אל תשנה `surface`; הוסף `normalized_surface` רק אם שונה
- [ ] E3 ודא שמספרים / pair לא יוצרים mentions

## F) LLM‑Rescue (אופציונלי)
- [ ] F1 `rescue_canonical_terms(surface_he)` → 1–3 EN terms (regex guard ^[A-Za-z][A-Za-z -]{0,40}$)
- [ ] F2 רוץ רק אם לטוקן word אין כלל lex + vec
- [ ] F3 הזרקת canonical_terms ושאילת vector נוספת (בלי יצירת IRI ישיר)
- [ ] F4 סימון metadata `notes:["llm_rescue"]`

## G) בדיקות יחידה
- [ ] G1 "מולטיוניט" → IRI OHD *dental implant abutment* `iri_source=resolved_alias`/`ohd_synonym`, `confident_singleton=True`
- [ ] G2 "שלת" → fuzzy→ OHD implant, `normalized_surface="שתל"`, score≥0.7
- [ ] G3 השבתת וקטור → score_final==score_lex (≥0.9)
- [ ] G4 מצב אין התאמות → Rescue מייצר canonical_terms ווקטור מחזיר ≥1 מועמד
- [ ] G5 מספרים/זוגות → 0 mentions

## H) ריצת אימות
- [ ] H1 דוגמת קלט `מולטיוניט שלת14 18/0` → שני mentions עם IRIs אמיתיים
- [ ] H2 לוג query_text נוכח + יש vector hits
- [ ] H3 ללא rescue אם כבר נמצאו התאמות

## Acceptance Criteria
- כל alias placeholder הוחלף או מסומן alias_only עם penalty
- שני ה־mentions בדוגמת הקלט מפיקים IRIs OHD + score_final>0
- fuzzy typo מקבל normalized_surface
- בדיקות G1–G5 ירוקות

## סדר ביצוע מומלץ
A → B → C → D → E → (F) → G → H

## הערות
- שמור שינויים קטנים וממוקדים בכל קובץ
- הימנע מכתיבת לוג INFO רועש (השתמש בDEBUG)
- ודא תאימות אחורה של פורמט JSON אם קיים צרכן downstream
