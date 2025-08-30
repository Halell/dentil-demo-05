# Block 2 – תוכנית תיקונים ומעקב ביצוע

(סמן V כשבוצע, או ~ אם חלקי)

## A) Alias → OHD Resolver
- [x] A1 פונקציה resolve_alias_to_ohd (חיפוש label / synonyms / וקטור קצר)
- [x] A2 שימוש ב-resolver בעת טעינת `en_alias_to_iri.json`
- [x] A3 ניסיון resolve גם ל-he2en_static (לאנגליות) לפני שמירת placeholder
- [x] A4 סימון alias_only=true כשלא נמצא IRI אמיתי (בוצע גם penalty + דידופין עדיף על placeholder)

## B) מערכת ניקוד / Hybrid Re-Weight
- [x] B1 ביטול מינ-מקס ללקסיקון (שימוש ישיר ב-score_lex)
- [x] B2 נרמול רק לוקטורים (norm_vec)
- [x] B3 Re-weight דינמי לפי רכיבים נוכחים
- [x] B4 fuzzy score = normalized similarity (0.70–0.85)
- [x] B5 confident singleton rule (score_lex≥0.9 & מועמד יחיד)

## C) שכבת וקטורים (FAISS Health)
- [x] C1 וידוא ENV / CONFIG נתיבים נכונים (טעינה דינמית + fallback)
- [x] C2 בדיקת index.ntotal>0 והדפסת לוג אם ריק
- [x] C3 התאמת מודל אמבדינג לערך meta['model_name']
- [x] C4 הרחבת build_query_text: surface/he2en/canonical + שכנים (לפני/אחרי)

## D) LLM‑Rescue (אופציונלי)
- [x] D1 פונקציה llm_canonicalize + Guardrails (JSON strict)
- [x] D2 אינטגרציה ב-router: רק אם אין lex ואין vec
- [x] D3 ריצת vector חוזרת עם canonical_terms_en
- [x] D4 תיעוד notes=["llm_rescue"]

## E) שיפורי Gazetteer משלימים
- [x] E1 Fast-path לטוקן יחיד + cover mask
- [x] E2 Damerau + norm_he (he_fuzzy_ok)
- [x] E3 עדכון ציון fuzzy לפי similarity (טווח משופר)
- [x] E4 מילוי covered_token_idxs ב-Hit (רשימת אינדקסים)
- [x] E5 Dedup post-resolve (בחירת IRI אמיתי על פני placeholder)

## F) סינון מספרים / זוגות / יחידות
- [x] F1 אי יצירת mentions לטוקנים kind∈{number,pair,unit}
- [x] F2 סינון גם בשכבת vector

## G) בדיקות יחידה
- [x] G1 multiunit alias → OHD IRI + confident_singleton
- [x] G2 shalat fuzzy → OHD implant IRI (score_lex≈0.75–0.85)
- [x] G3 vector-only→rescue scenario
- [x] G4 no vector index → score_final == score_lex
- [x] G5 numbers only → zero mentions
- [x] G6 re-weight without vec (score_final==score_lex)

## H) אימות אינטגרטיבי
- [ ] H1 ריצה על הדוגמה ושמירת merged עם IRIs אמיתיים
- [ ] H2 בדיקת log שאין אזהרות טעינת אינדקס
- [ ] H3 מדד basic: זמן ריצה לשורה < 1s (ללא LLM)

## I) תיעוד / ניקיון
- [x] I1 עדכון README (סעיף Block2 scoring & rescue)
- [x] I2 הוספת gitignore לפריטים נוספים (cache / artifacts logs)
- [x] I3 הערות TODO ל-Block 2.5 (Pruner / Bundles / Context)

### TODO Block 2.5 (Outline)
- Pruner: remove low-score vector distractors sharing surface subset.
- Bundles: group adjacent device + size measurements.
- Context scoring: upweight candidates with consistent material/device hints.
- Neo4j backend parity & latency benchmarking.

---

### מצב נוכחי מקוצר
- A–F הושלמו (כולל Rescue אופציונלי + Dedup E5 + fuzzy scaling E3)
- נותר: G (בדיקות), H (אימות אינטגרטיבי), I (תיעוד וניקיון)

