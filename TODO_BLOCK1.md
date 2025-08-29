# משימות להשלמת בלוק 1 - מערכת OHD Clinic NLP

## סקירה כללית
בלוק 1 מתמקד בעיבוד דטרמיניסטי של טקסט דנטלי עם הכנה מושלמת לשלבים הבאים.
- **N0** - נירמול דטרמיניסטי (איבוד-אפס)  
- **T1** - טוקניזציה דטרמיניסטית
- **N0b** - עידון נירמול בעזרת LLM (אופציונלי, בטוח)
- **T1b** - תיוגי עזר/רמזים בעזרת LLM

## ✅ רשימת משימות מפורטת

### 1. עדכון מבנה הפרויקט
**תיקיות וקבצים חדשים לייצר:**
```
ohd-clinic-nlp/
├─ data/
│  ├─ dictionaries/
│  │  └─ tooth_groups.json
│  └─ samples/lines.txt
├─ artifacts/runs/block1/
├─ src/
│  ├─ pipeline/
│  │  ├─ n0_normalize.py
│  │  ├─ t1_tokenize.py
│  │  ├─ mv_builder.py
│  │  ├─ n0b_llm_refine.py
│  │  ├─ t1b_llm_hints.py
│  │  └─ validators.py
│  ├─ llm/prompts/
│  │  └─ n0b_t1b_prompt.txt
│  ├─ common/
│  │  └─ schemas.py (לעדכן)
│  └─ cli/
│     └─ block1.py
└─ tests/unit/
   ├─ test_n0_normalize.py
   ├─ test_t1_tokenize.py
   ├─ test_llm_invariants.py
   └─ test_cli_block1.py
```

### 2. N0 - נירמול דטרמיניסטי
**קובץ:** `src/pipeline/n0_normalize.py`

**משימות פיתוח:**
- [ ] יישום פונקציה `normalize_n0(raw_text: str) -> NormalizationResult`
- [ ] הוספת רווח בין אותיות לספרות
- [ ] זיהוי זוגות מספרים עם מפרידים (`/`, `\`, `-`)
- [ ] נירמול יחידות קיימות (mm, מ"מ, °)
- [ ] יישום NFC/NFKC ל-Unicode
- [ ] שמירת raw_text ו-spans

**כללי איכות:**
- אין שינוי משמעות
- אין הוספת יחידות שלא הופיעו
- שמירת סדר מספרים

### 3. T1 - טוקניזציה דטרמיניסטית  
**קובץ:** `src/pipeline/t1_tokenize.py`

**משימות פיתוח:**
- [ ] יישום `tokenize_t1(n0: NormalizationResult) -> TokensResult`
- [ ] זיהוי pairs כטוקן יחיד עם meta
- [ ] חלוקה לטוקנים: number, word, unit, punct
- [ ] טיפול במקפים פנימיים במילים
- [ ] חישוב spans מדויק לכל טוקן
- [ ] תמיכה בעברית ואנגלית (script detection)

### 4. Marked View Builder
**קובץ:** `src/pipeline/mv_builder.py`

**משימות פיתוח:**
- [ ] יישום `build_marked_view(n0, t1) -> str`
- [ ] יצירת LEGEND עם הגדרות וכללים
- [ ] פירמוט TOKENS עם אינדקסים וspans
- [ ] הצגת PAIRS מפורטת
- [ ] הוספת INSTRUCTIONS לLLM
- [ ] כללי בטיחות מפורשים

### 5. N0b - עידון נירמול עם LLM
**קובץ:** `src/pipeline/n0b_llm_refine.py`

**משימות פיתוח:**
- [ ] יישום `refine_with_llm(mv_text: str) -> dict`
- [ ] פיתוח פרומפט מדויק עם כללי בטיחות
- [ ] טיפול בoperations: insert_space, merge_tokens
- [ ] החזרת canonical_terms, tooth_groups, intent_hints
- [ ] יישום ולידציה קפדנית על התוצאה
- [ ] טיפול בambiguous cases

### 6. T1b - תיוגי עזר עם LLM
**קובץ:** `src/pipeline/t1b_llm_hints.py`

**משימות פיתוח:**
- [ ] הוספת תגים לטוקנים: kw_device_hint, kw_implant_hint
- [ ] זיהוי tooth_groups מהמילון המאושר
- [ ] הוספת intent_hints
- [ ] שמירה על אינווריאנטים (אין שינוי מספרים)

### 7. ולידטורים
**קובץ:** `src/pipeline/validators.py`

**משימות יישום:**
- [ ] בדיקת שמירת רשימת מספרים
- [ ] בדיקת שמירת pairs (A, B, sep)
- [ ] בדיקת כיסוי תווים
- [ ] אימות tooth_groups מול מילון
- [ ] החזרת שגיאות מפורטות

### 8. CLI - ממשק הרצה
**קובץ:** `src/cli/block1.py`

**משימות פיתוח:**
- [ ] פקודת `run-n0-t1` לעיבוד בסיסי
- [ ] פקודת `run-all` עם LLM
- [ ] קריאת קובץ קלט (lines.txt)
- [ ] כתיבת פלטים ל-artifacts/runs/block1/:
  - n0_normalized.jsonl
  - t1_tokens.jsonl  
  - n0b_t1b_llm_aug.jsonl
  - merged_block1.jsonl

### 9. מילון קבוצות שיניים
**קובץ:** `data/dictionaries/tooth_groups.json`

**תוכן לייצר:**
```json
{
  "lower_incisors": {"label_he": "חותכות תחתונות", "FDI": ["31","32","41","42"]},
  "upper_incisors": {"label_he": "חותכות עליונות", "FDI": ["11","12","21","22"]},
  "lower_canines": {"label_he": "ניבים תחתונים", "FDI": ["33","43"]},
  "upper_canines": {"label_he": "ניבים עליונים", "FDI": ["13","23"]},
  "upper_premolars": {"label_he": "טוחנות קטנות עליונות", "FDI": ["14","15","24","25"]},
  "lower_premolars": {"label_he": "טוחנות קטנות תחתונות", "FDI": ["34","35","44","45"]},
  "upper_molars": {"label_he": "טוחנות עליונות", "FDI": ["16","17","18","26","27","28"]},
  "lower_molars": {"label_he": "טוחנות תחתונות", "FDI": ["36","37","38","46","47","48"]}
}
```

### 10. סכימות Pydantic
**קובץ:** `src/common/schemas.py`

**סכימות להוסיף:**
- [ ] NormalizationResult (raw_text, normalized_text, numbers, pairs, units_found)
- [ ] Pair (text, A, B, sep, span)
- [ ] Token (idx, text, kind, span, script, meta)
- [ ] TokensResult (text, tokens)
- [ ] LlmAugmentResult (ops, canonical_terms, tooth_groups, intent_hints, ambiguous)

### 11. פרומפט LLM
**קובץ:** `src/llm/prompts/n0b_t1b_prompt.txt`

**אלמנטים חובה:**
- הנחיות בטיחות (אסור לשנות מספרים/pairs)
- פורמט JSON מחייב
- דוגמאות
- תנאי ambiguous

### 12. בדיקות יחידה

#### test_n0_normalize.py
- [ ] בדיקת הוספת רווחים נכונה
- [ ] בדיקת זיהוי pairs
- [ ] בדיקת אי-שינוי ספרות
- [ ] בדיקת נירמול יחידות

#### test_t1_tokenize.py  
- [ ] בדיקת pair כטוקן יחיד
- [ ] בדיקת כיסוי spans
- [ ] בדיקת מילים עם מקף פנימי
- [ ] בדיקת חלוקה נכונה לסוגי טוקנים

#### test_llm_invariants.py
- [ ] בדיקת דחיית שינוי pairs
- [ ] בדיקת קבלת ops בטוחות
- [ ] בדיקת tooth_groups מול מילון

#### test_cli_block1.py
- [ ] בדיקת יצירת קבצי פלט
- [ ] בדיקת JSON תקין
- [ ] בדיקת pipeline מלא

### 13. קבצי דוגמה
**קובץ:** `data/samples/lines.txt`

**דוגמאות לכלול:**
```
מולטיוניט שתל14 18/0
חותכות תחתונות - ביקורת
כתר על שן 36 עם מרווח 2mm
טיפול שורש 14-16 דחוף
```

## 🎯 Acceptance Criteria - הגדרת DONE

### קריטריונים טכניים:
- [ ] כל הקבצים נוצרו במיקומים הנכונים
- [ ] n0_normalize.py עובד ועובר בדיקות
- [ ] t1_tokenize.py עובד ועובר בדיקות  
- [ ] mv_builder.py מייצר MV תקני
- [ ] LLM modules מחזירים JSON תקין עם ולידציה
- [ ] CLI run-all יוצר 4 קבצי פלט
- [ ] כל הבדיקות עוברות בירוק

### קריטריונים פונקציונליים:
- [ ] אין הפרת אינווריאנטים (מספרים ו-pairs נשמרים)
- [ ] זיהוי נכון של זוגות מספרים
- [ ] טוקניזציה נכונה עם spans מדויקים
- [ ] LLM מחזיר canonical_terms רלוונטיים
- [ ] tooth_groups מזוהות נכון מהמילון
- [ ] intent_hints מתאימים להקשר

### דוגמאות בדיקה:

**קלט:** `מולטיוניט שתל14 18/0`
- N0: הוספת רווח → `מולטיוניט שתל 14 18/0`
- T1: 5 טוקנים כולל pair `18/0`
- LLM: `canonical_terms=["multi-unit abutment","dental implant"]`

**קלט:** `חותכות תחתונות - ביקורת`
- LLM: `tooth_groups` עם FDI `[31,32,41,42]`
- LLM: `intent_hints=["scheduling"]`

## 📝 הערות חשובות

### ביצועים:
- N0/T1 צריכים לרוץ ב-ms
- קריאת LLM רק כשיש צורך אמיתי
- regex מקומפלים מראש

### בטיחות:
- אסור שינוי מספרים/pairs
- JSON Schema validation חובה
- fallback ל-ambiguous=true במקרי ספק

### Unicode ועברית:
- נירמול ל-NFC
- spans לפי תווים (לא bytes)
- תמיכה דו-כיוונית (HE/EN)

## 🚀 סדר ביצוע מומלץ

1. **שלב א':** מבנה פרויקט + סכימות
2. **שלב ב':** N0 + T1 + בדיקות
3. **שלב ג':** MV Builder + Validators
4. **שלב ד':** LLM modules (N0b/T1b)
5. **שלב ה':** CLI + בדיקות אינטגרציה
6. **שלב ו':** הרצת בדיקות מלאות + תיקונים

---

**מוכן להתחיל? כל משימה מוגדרת היטב ומוכנה ליישום.**