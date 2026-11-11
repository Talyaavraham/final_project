# Video Search Agent — Backend (FastAPI + YOLOv8 + CLIP + Claude)

## התקנה והרצה

```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows   (במאק/לינוקס: source venv/bin/activate)
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

בהרצה הראשונה ultralytics מוריד אוטומטית את המשקולות `yolov8n.pt` (~6MB).
אם CLIP מופעל (ברירת מחדל), בחיפוש הראשון יורדים גם משקולות CLIP (~600MB, פעם אחת).

### משתני סביבה

| משתנה | ברירת מחדל | תיאור |
|---|---|---|
| `YOLO_MODEL` | `yolov8n.pt` | `yolov8s.pt` / `yolov8m.pt` לדיוק גבוה יותר (GPU מומלץ). `yolov8s-worldv2.pt` = **YOLO-World**, זיהוי open-vocabulary של רשימת המחלקות ב-`indexer.OPEN_VOCAB` (עגלת תינוק, קורקינט, כובע, קסדה, טלפון…) |
| `OPEN_VOCAB` | — | רשימה מופרדת בפסיקים שמחליפה את `indexer.OPEN_VOCAB` עבור YOLO-World, למשל `tree,apple,character` |
| `YOLO_MODEL=game.pt` | — | מודל שאומן על המחלקות שלכן (דמות, עץ, פרי, שמש במשחק של ex4_demo; mAP50 0.97). איך מאמנים: [train/README.md](train/README.md). להריץ עם `IMGSZ=800` |
| `YOLO_MODEL=yolov8n.pt,game.pt` | — | **כמה מודלים**: לכל סרטון נבחר המודל שמוצא הכי הרבה ב-20 פריימים מפוזרים (סכום הביטחון של הזיהויים). סרטון מהמציאות מקבל את מודל COCO, המשחק מקבל את `game.pt`. המודל שנבחר נשמר ב-`/videos` |
| `EXTRA_MODELS=security.pt` | — | מודלים שרצים על כל פריים **בנוסף** למודל שנבחר לסרטון. `security.pt` (yolov8s, בריפו): gun / knife / fire / smoke, אומן על CCTV weapons + D-Fire; mAP50 על validation: gun 0.69, knife 0.35, fire 0.73, smoke 0.84. אימון: `train/security_dataset.py`, `train/colab_train_security.ipynb` |
| `EXTRA_CONF` | `0.2` | סף הביטחון למודלים הנוספים (אירועים נדירים: עדיף לשמור זיהוי חלש; track עדיין צריך 2+ זיהויים) |
| `YOLO_MODEL=yolov8n.pt,game.pt,night.pt` | — | `night.pt` = מודל שאומן על תמונות לילה (ExDark + פריימים שלכן) לסרטוני מצלמות אבטחה חשוכים. איך: [train/README.md](train/README.md), `train/night_dataset.py`, `train/colab_train_night.ipynb` |
| `YOLO_MODEL=yolov8s-oiv7.pt` | — | Open Images V7, 601 מחלקות (Tree, Window, Footwear…). פחות אמין מ-COCO על אנשים ורכבים, אז לא ברירת מחדל. Man/Woman/Boy/Girl ממופים ל-`person`, חלקי גוף (Human face וכו') מדולגים |
| `CLIP_RERANK` | `1` | `0` מכבה את CLIP (חיפוש סמנטי בטקסט חופשי + מיזוג זהויות) |
| `ANTHROPIC_API_KEY` | — | כשמוגדר, השאילתות מפוענחות ע"י Claude (`claude-opus-5-5`) במקום חוקי מילון |
| `SAMPLE_FPS` | `3` | כמה פריימים לשנייה מנותחים |
| `CONF_THRESHOLD` | `0.3` | סף ביטחון לזיהוי |
| `MIN_SCENE` / `SCENE_Z` | `0.2` / `1.5` | חיפוש סצנה (CLIP על הפריים השלם, כל שנייה). `MIN_SCENE` הוא **הסתברות**: softmax של השאילתה (בארבעה ניסוחים) מול פרומפטים של "שום דבר מיוחד" (`search.match_prob`), ולא דמיון קוסינוס גולמי. `SCENE_Z`: כמה סטיות תקן מעל הממוצע של הסרטון. נמדד עם `eval/sweep_scene.py` |
| `TRACK_VIEWS` | `4` | כמה מבטים עם הקשר (ריבוע סביב התיבה, פי 1.3) נשמרים ל-CLIP לכל track, בנוסף לחיתוך הטוב ביותר. `0` = רק החיתוך הטוב ביותר (שלב ה-CLIP מהיר בערך פי 2) |
| `SCENE_EVERY` | `1.0` | כל כמה שניות נשמר וקטור CLIP של הפריים השלם (`0` מכבה חיפוש סצנה) |
| `MIN_SEMANTIC` / `MIN_SEMANTIC_UNKNOWN` | `0.20` / `0.27` | סף הדמיון של CLIP ל-track (ממוצע על המבטים שלו). הסף הגבוה חל כשהשאילתה מכילה מילים שאין להן מחלקה בגלאי ("knife", "gun", "package"): אז CLIP הוא הראיה היחידה. כויל עם `eval/offline_eval.py` |
| `IMGSZ` | `640` | גודל הקלט לגלאי. `800` לאובייקטים קטנים (הפירות במשחק הם ~24 פיקסלים ב-1080p) |
| `VIDEO_DIR` / `DB_PATH` / `EXPORT_DIR` | `videos` / `index.db` / `exports` | מיקומי קבצים |
| `CORS_ORIGINS` | `*` | להגביל בפרודקשן |

### Docker

```bash
docker build -t video-search backend
docker run -p 8000:8000 -v $(pwd)/data:/data -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY video-search
```

## איך זה מתחבר לפרונטאנד

1. מריצים את השרת ופותחים **http://localhost:8000** (השרת מגיש את `index.html`)
2. מעלים סרטון — הפרונט שולח ל-`POST /index`. השרת מחשב hash של הקובץ:
   סרטון שכבר אונדקס (גם בשם אחר) לא מאונדקס שוב
3. השאלה נשלחת ל-`POST /parse` (Claude או חוקים) ואז ל-`POST /search`
4. כל תוצאה ניתנת להורדה כתמונה (`/frame`) או קליפ (`/clip`), וכל התוצאות יחד כ-ZIP (`/export`)
5. אם השרת לא רץ — הפרונט מזהה בדפדפן (COCO-SSD) ומוריד תמונות דרך canvas

## Endpoints

| Method | Path | תיאור |
|---|---|---|
| GET | `/` | הממשק (index.html) |
| GET | `/api` | יכולות השרת: מודל, open_vocab, clip, llm_parser, ffmpeg |
| POST | `/index` (multipart, `?force=1`) | אינדוקס ברקע. מחזיר `cached: true` אם התוכן כבר מאונדקס |
| GET | `/status?video=` | `indexing` / `ready` / `error`, עם `progress` ו-`stage` |
| GET | `/videos`, DELETE `/videos/{name}` | רשימת סרטונים / מחיקה |
| GET | `/classes?video=` | אילו מחלקות נמצאו בסרטון (וכמה) |
| POST | `/parse` `{text, video}` | טקסט חופשי → `{objects, colors, actions, timeRange, semantic, unknown, source}` |
| POST | `/search` `{video, query, limit, offset, near}` | → `{total, results[]}` עם pagination |
| GET | `/frame?video&t&x&y&w&h&label` | JPEG של הפריים עם המסגרת |
| GET | `/clip?video&t0&t1&track&annotate` | MP4 של הקטע, עם מסגרת עוקבת כש-`track` נתון |
| POST | `/export` `{video, matches, query, clips, annotate, pad}` | ZIP: `frames/*.jpg`, `clips/*.mp4`, `results.json`, `results.csv` |

## מה יש בפנים

```
main.py          API (FastAPI + Pydantic)
indexer.py       YOLO/YOLO-World + ByteTrack → צבע → כתיבה ל-SQLite במנות → segments → מעבר שני: CLIP (פריימים + מבטים של כל track) → מיזוג זהויות.  `python indexer.py reembed` מריץ מחדש רק את שלב ה-CLIP
search.py        segments, מהירות/פעולות, קרבה במרחב, דירוג סמנטי, חיפוש סצנה, `run_query` = כל התשובה לשאילתה   (לוגיקה טהורה, נבדקת ביחידה)
query_parser.py  Claude (structured output) + מפענח חוקים EN/HE כגיבוי
embeddings.py    CLIP (transformers), אופציונלי
export.py        פריימים, קליפים (ffmpeg מובנה דרך imageio-ffmpeg), ZIP
db.py            סכמת SQLite: videos / detections / segments
evaluate.py      precision / recall / F1 מול ground truth מתויג (eval/: benchmark UCF-Crime, כיול ספים)
train/           תיוג אוטומטי לפי צבע + אימון YOLOv8 על מחלקות חדשות (train/README.md)
tests/           pytest — 51 בדיקות, כולל end-to-end עם גלאי מדומה ו-CLIP מדומה
```

- **זיהוי**: כל 80 מחלקות COCO (לא רק 9), או רשימה פתוחה עם YOLO-World
- **מעקב**: ByteTrack (`model.track(persist=True)`)
- **צבע**: הצבעה של פיקסלים על שם צבע ב-HSV (לאדם: החלק העליון = חולצה)
- **פעולות**: מהירות המרכז בגבהי-גוף לשנייה — running / walking / parked, ו-entering / exiting לפי קצה הפריים
- **"איש עם כלב"**: חפיפה בזמן **וגם** קרבה במרחב (`NEAR_DIST`, ניתן לכבות עם `near:false`)
- **חיפוש סמנטי**: כל track מקבל כמה וקטורי CLIP (החיתוך הטוב ביותר + מבטים עם הקשר לאורך חייו); טקסט חופשי ("a woman pushing a stroller") מדורג לפי דמיון קוסינוס ממוצע
- **חיפוש סצנה**: וקטור CLIP של הפריים כל שנייה (בפריים רחב, 16:9: כמה ריבועים שמכסים את כולו), מדורג ב-softmax מול פרומפטים ניטרליים
- **גלאי + CLIP יחד**: למחלקות אירוע (אש, עשן, אקדח, סכין) ה-tracks של הגלאי נשארים כמו שהם ו-CLIP מוסיף את הרגעים שהגלאי פספס. מחלקת אירוע שלא נמצאה בסרטון (או שהסרטון אונדקס בלי `security.pt`) הופכת לשאלת CLIP; מחלקת COCO שלא נמצאה ("כלב") נשארת תשובה ריקה
- **עברית בלי Claude**: CLIP מבין אנגלית בלבד, אז מפענח החוקים בונה לו משפט אנגלי מהמילים שזוהו ("אדם עם סכין" → "a person with a knife")
- **מיזוג זהויות**: tracks מאותה מחלקה עם embedding דומה שלא נראים יחד בזמן → אותו `entity`
- **סרטונים ארוכים**: הזיהויים נכתבים ל-DB כל 2000 שורות; ה-segments מחושבים פעם אחת באינדוקס, לא בכל חיפוש
- **מגבלה ידועה**: זיהוי הפעולות מניח מצלמה סטטית. במצלמה נעה, "running" יכול להופיע גם לאובייקטים עומדים

## בדיקות והערכה

```bash
pytest tests -q                       # בדיקות יחידה + end-to-end (ללא משקולות)
python evaluate.py eval/ground_truth.json --api http://localhost:8000
python evaluate.py eval/ground_truth_uca.json --match hit   # התרעות: קטע בתוך חלון = פגיעה
```

`eval/ground_truth.example.json` מראה את הפורמט: לכל סרטון רשימת שאילתות וחלונות הזמן
הנכונים. הסקריפט מדפיס precision / recall / F1 לכל שאילתה, לכל תרחיש ובסך הכול, וכותב `<ground_truth>_report.json`.
`--match iou` (ברירת מחדל) דורש חפיפה של 30% בין הקטע לחלון; `--match hit` סופר קטע שנמצא בתוך חלון (התרעה נכונה).

**Benchmark אבטחה (UCF-Crime + UCA):** `eval/uca_ground_truth.py` בונה ground truth מהמשפטים המתוייגים
של UCA (סכין, אקדח, אש, חטיפת תיק, חבילה) ל-107 סרטונים; `eval/run_benchmark.ps1` מאנדקס אותם
ומריץ את ההערכה דרך השרת.

**בלי שרת ובלי לאנדקס מחדש:**

```bash
python eval/offline_eval.py eval/benchmark.db eval/ground_truth_uca.json          # אותו קוד כמו /search, שניות
python eval/offline_eval.py ... --set MIN_SCENE=0.3 --track-views 1                # כיול ספים / ablation
python eval/offline_eval.py ... --extra-raw eval/results/detector_security_raw_2026-11-10.json   # + ה-tracks ש-security.pt היה מוסיף
python indexer.py reembed --db eval/benchmark.db --videos eval/data/ucf           # רק שלב ה-CLIP מחדש (הזיהויים נשארים)
```

תוצאות (hit scoring, 189 חלונות, `eval/results/`):

| | F1 | precision | recall |
|---|---|---|---|
| 2026-11-07, דרך השרת | 0.164 | 0.11 | 0.32 |
| לפני התיקונים של 2026-11-11 ("fire" החזיר 0 תוצאות) | 0.151 | 0.10 | 0.28 |
| 2026-11-11, אותו אינדקס, קוד חיפוש חדש | 0.178 | 0.12 | 0.35 |
| 2026-11-11, אינדקס עם כמה מבטים לכל track | 0.180 | 0.12 | 0.35 |
| + ה-tracks של `security.pt` (סימולציה מ-1 fps, בלי תיבות) | 0.186 | 0.12 | 0.39 |

מה נמדד ומה לא: השיפור בא מאש (F1 0 → 0.36–0.39; עם הגלאי 20 מתוך 44 חלונות). המבטים עם ההקשר משפרים את
**דירוג** ה-tracks של "אדם עם אקדח" (average precision 0.24 → 0.28) אבל לא את ה-F1 הכולל; ההבדלים בין
התרחישים האחרים קטנים מהרעש (שני חצאי הסרטונים נותנים 0.14 ו-0.28). סכין, חטיפה וחבילה נשארים קרוב לניחוש:
ב-320x240 אדם הוא כ-30 פיקסלים ו-CLIP ViT-B/32 לא רואה מה יש לו ביד. כיסוי הפריים בריבועים לא עזר ב-4:3
ולכן פועל רק בפריים רחב (לא נמדד: אין ב-benchmark סרטונים רחבים).
