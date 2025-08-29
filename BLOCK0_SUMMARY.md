# סיכום בלוק 0 (הכנה)

## מה הוקם
- מבנה תיקיות: data/ontology, artifacts/{lexicon,vectors}, src/{common,llm,ontology,tools}, models/embeddings, tests.
- קבצי בסיס: requirements.txt, .env / .env.example, README.md, ollama_client, build_lexicon, build_embeddings, sanity_checks, config, logging.
- הופק לקסיקון (3168 רשומות) + אינדקס FAISS (dim=384, count=3168).
- חיבור Ollama (כולל תמיכת Turbo + headers דינמיים).

## החלטות / בחירות
- מודל אמבדינג: fallback ל-`sentence-transformers/all-MiniLM-L6-v2` כי נתיב מקומי `bge-m3` לא קיים עדיין.
- נורמליזציה + IndexFlatIP (קוסינוס באמצעות normalizing) – בחירה פשוטה ומהירה לשלב זה.
- שמירת meta (`ohd_meta.json`) עם כל הרשומות (פשטות על חשבון משקל; ניתן לצמצם בהמשך).
- הסרת מפתחות/API ישנים מה-`.env` לשמירת סודות מחוץ ל-repo.
- הרחבת עטיפת Ollama לתמיכה ב-Turbo והפרדת headers (API_KEY מול TURBO_KEY) במקום לקודד ישירות host יחיד.
- שדה EMBED_DIM נשאר מוצהר ב-.env אך נגזר בפועל מהמודל (כרגע 384) – לא נכפה מספר כדי לאפשר החלפה גמישה.

## תקלות ופתרונות
- Path למודל אמבדינג מקומי לא נמצא → נוספה רשימת fallback וטעינה דינמית.
- ניסיון ראשון לבנות אמבדינגס עם שם מודל לא ציבורי (bge-m3) → כשל 401 HF → מעבר לפומבי MiniLM.
- קובץ `ohd.owl` היה בשורש → הוכנס ל-`data/ontology/` וריבילד.
- Placeholder ל-`ohd.owl` גרם לשגיאת SAX (no element found) → הוחלף בקובץ המלא.
- JSON parsing קשיח מעטיפת Ollama: הוספה עטיפה המחזירה {"raw": ...} אם parse נכשל.

## סטטוס סופי
כל סעיפי ההגדרה ל-Done של בלוק 0 סגורים. המערכת מוכנה לבלוק הבא (אנוטציה/פירוק/קישור).
