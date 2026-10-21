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
| `CLIP_RERANK` | `1` | `0` מכבה את CLIP (חיפוש סמנטי בטקסט חופשי + מיזוג זהויות) |
| `ANTHROPIC_API_KEY` | — | כשמוגדר, השאילתות מפוענחות ע"י Claude (`claude-opus-5-5`) במקום חוקי מילון |
| `SAMPLE_FPS` | `3` | כמה פריימים לשנייה מנותחים |
| `CONF_THRESHOLD` | `0.3` | סף ביטחון לזיהוי |
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
indexer.py       YOLO/YOLO-World + ByteTrack → צבע → כתיבה ל-SQLite במנות → segments → CLIP → מיזוג זהויות
search.py        segments, מהירות/פעולות, קרבה במרחב, דירוג סמנטי   (לוגיקה טהורה, נבדקת ביחידה)
query_parser.py  Claude (structured output) + מפענח חוקים EN/HE כגיבוי
embeddings.py    CLIP (transformers), אופציונלי
export.py        פריימים, קליפים (ffmpeg מובנה דרך imageio-ffmpeg), ZIP
db.py            סכמת SQLite: videos / detections / segments
evaluate.py      precision / recall / F1 מול ground truth מתויג
train/           תיוג אוטומטי לפי צבע + אימון YOLOv8 על מחלקות חדשות (train/README.md)
tests/           pytest — 23 בדיקות, כולל end-to-end עם גלאי מדומה
```

- **זיהוי**: כל 80 מחלקות COCO (לא רק 9), או רשימה פתוחה עם YOLO-World
- **מעקב**: ByteTrack (`model.track(persist=True)`)
- **צבע**: הצבעה של פיקסלים על שם צבע ב-HSV (לאדם: החלק העליון = חולצה)
- **פעולות**: מהירות המרכז בגבהי-גוף לשנייה — running / walking / parked, ו-entering / exiting לפי קצה הפריים
- **"איש עם כלב"**: חפיפה בזמן **וגם** קרבה במרחב (`NEAR_DIST`, ניתן לכבות עם `near:false`)
- **חיפוש סמנטי**: כל track מקבל embedding של CLIP מהחיתוך הטוב ביותר; טקסט חופשי ("a woman pushing a stroller") מדורג לפי דמיון קוסינוס
- **מיזוג זהויות**: tracks מאותה מחלקה עם embedding דומה שלא נראים יחד בזמן → אותו `entity`
- **סרטונים ארוכים**: הזיהויים נכתבים ל-DB כל 2000 שורות; ה-segments מחושבים פעם אחת באינדוקס, לא בכל חיפוש
- **מגבלה ידועה**: זיהוי הפעולות מניח מצלמה סטטית. במצלמה נעה, "running" יכול להופיע גם לאובייקטים עומדים

## בדיקות והערכה

```bash
pytest tests -q                       # בדיקות יחידה + end-to-end (ללא משקולות)
python evaluate.py eval/ground_truth.json --api http://localhost:8000
```

`eval/ground_truth.example.json` מראה את הפורמט: לכל סרטון רשימת שאילתות וחלונות הזמן
הנכונים. הסקריפט מדפיס precision / recall / F1 לכל שאילתה ובסך הכול, וכותב `eval/report.json`.
