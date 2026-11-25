# Video Search Agent — Backend (FastAPI + YOLOv8 + CLIP)

הכול רץ מקומית, על המודלים שלנו: לא נשלח שום דבר לשירות חיצוני.

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
| `EXTRA_MODELS=security.pt` | — | מודלים שרצים על כל פריים **בנוסף** למודל שנבחר לסרטון. `security.pt` (yolov8s, בריפו): gun / knife / fire / smoke, אומן על CCTV weapons + D-Fire; mAP50 על validation: gun 0.69, knife 0.35, fire 0.73, smoke 0.84. הסכין חלש (210 תיבות אימון; 0 זיהויים על 585 תמונות סכין חדשות). אימון שני עם 3,600 תיבות סכין מצילומים קרובים (`security_v2.pt`, 2026-11-23, לא בריפו) מזהה את התמונות האלה (mAP50 0.95) אבל לא עוזר ב-UCF-Crime, ולכן לא החליף את `security.pt`: ראו "אימון שני של הגלאי" למטה. אימון: `train/security_dataset.py`, `train/colab_train_security.ipynb`, פירוט ב-`train/README.md` |
| `EXTRA_CONF` | `0.2` | סף הביטחון למודלים הנוספים (אירועים נדירים: עדיף לשמור זיהוי חלש; track עדיין צריך 2+ זיהויים) |
| `YOLO_MODEL=yolov8n.pt,game.pt,night.pt` | — | `night.pt` = מודל שאומן על תמונות לילה (ExDark + פריימים שלכן) לסרטוני מצלמות אבטחה חשוכים. איך: [train/README.md](train/README.md), `train/night_dataset.py`, `train/colab_train_night.ipynb` |
| `YOLO_MODEL=yolov8s-oiv7.pt` | — | Open Images V7, 601 מחלקות (Tree, Window, Footwear…). פחות אמין מ-COCO על אנשים ורכבים, אז לא ברירת מחדל. Man/Woman/Boy/Girl ממופים ל-`person`, חלקי גוף (Human face וכו') מדולגים |
| `CLIP_RERANK` | `1` | `0` מכבה את CLIP (חיפוש סמנטי בטקסט חופשי + מיזוג זהויות) |
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
docker run -p 8000:8000 -v $(pwd)/data:/data video-search
```

## איך זה מתחבר לפרונטאנד

1. מריצים את השרת ופותחים **http://localhost:8000** (השרת מגיש את `index.html`)
2. מעלים סרטון — הפרונט שולח ל-`POST /index`. השרת מחשב hash של הקובץ:
   סרטון שכבר אונדקס (גם בשם אחר) לא מאונדקס שוב
3. השאלה נשלחת ל-`POST /parse` (מפענח מקומי) ואז ל-`POST /search`
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
| POST | `/parse` `{text, video}` | טקסט חופשי → `{objects, colors, actions, timeRange, minDuration, semantic, unknown, source}` |
| POST | `/search` `{video, query, limit, offset, near}` | → `{total, results[]}` עם pagination |
| GET | `/frame?video&t&x&y&w&h&label` | JPEG של הפריים עם המסגרת |
| GET | `/clip?video&t0&t1&track&annotate` | MP4 של הקטע, עם מסגרת עוקבת כש-`track` נתון |
| POST | `/export` `{video, matches, query, clips, annotate, pad}` | ZIP: `frames/*.jpg`, `clips/*.mp4`, `results.json`, `results.csv` |

## מה יש בפנים

```
main.py          API (FastAPI + Pydantic)
indexer.py       YOLO/YOLO-World + ByteTrack → צבע → כתיבה ל-SQLite במנות → segments → מעבר שני: CLIP (פריימים + מבטים של כל track) → מיזוג זהויות.  `python indexer.py reembed` מריץ מחדש רק את שלב ה-CLIP
search.py        segments, מהירות/פעולות, קרבה במרחב, דירוג סמנטי, חיפוש סצנה, `run_query` = כל התשובה לשאילתה   (לוגיקה טהורה, נבדקת ביחידה)
query_parser.py  מפענח חוקים EN/HE (מקומי)
embeddings.py    CLIP (transformers), אופציונלי
export.py        פריימים, קליפים (ffmpeg מובנה דרך imageio-ffmpeg), ZIP
db.py            סכמת SQLite: videos / detections / segments
evaluate.py      precision / recall / F1 מול ground truth מתויג (eval/: benchmark UCF-Crime, כיול ספים)
train/           תיוג אוטומטי לפי צבע + אימון YOLOv8 על מחלקות חדשות (train/README.md)
tests/           pytest — 95 בדיקות, כולל end-to-end עם גלאי מדומה ו-CLIP מדומה
```

- **זיהוי**: כל 80 מחלקות COCO (לא רק 9), או רשימה פתוחה עם YOLO-World
- **מעקב**: ByteTrack (`model.track(persist=True)`)
- **צבע**: הצבעה של פיקסלים על שם צבע ב-HSV (לאדם: החלק העליון = חולצה)
- **פעולות**: מהירות המרכז בגבהי-גוף לשנייה — running / walking / parked, ו-entering / exiting לפי קצה הפריים
- **כיוון, אזור ומשך בטקסט חופשי** (המסננים של מערכות כמו BriefCam, בלי תפריטים): "רכב נוסע שמאלה" = מרכז התיבה זז
  לפחות 10% מרוחב הפריים בכיוון הזה (`search.heading`; אלכסון נספר לשני הצירים); "אדם בצד ימין" / "למעלה" / "במרכז" =
  לפחות 30% מהזיהויים שלו באזור (`search.ZONES`); "עומד יותר מ-30 שניות" / "at least 2 minutes" = `minDuration`, משך ה-track.
  המשך נמדד ל-track אחד: אדם שהמעקב איבד וחזר נספר כשני קטעים קצרים
- **"איש עם כלב"**: חפיפה בזמן **וגם** קרבה במרחב (`NEAR_DIST`, ניתן לכבות עם `near:false`)
- **חיפוש סמנטי**: כל track מקבל כמה וקטורי CLIP (החיתוך הטוב ביותר + מבטים עם הקשר לאורך חייו); טקסט חופשי ("a woman pushing a stroller") מדורג לפי דמיון קוסינוס ממוצע
- **חיפוש סצנה**: וקטור CLIP של הפריים כל שנייה (בפריים רחב, 16:9: כמה ריבועים שמכסים את כולו), מדורג ב-softmax מול פרומפטים ניטרליים
- **גלאי + CLIP יחד**: למחלקות אירוע (אש, עשן, אקדח, סכין) ה-tracks של הגלאי נשארים כמו שהם ו-CLIP מוסיף את הרגעים שהגלאי פספס. מחלקת אירוע שלא נמצאה בסרטון (או שהסרטון אונדקס בלי `security.pt`) הופכת לשאלת CLIP; מחלקת COCO שלא נמצאה ("כלב") נשארת תשובה ריקה
- **עברית**: CLIP מבין אנגלית בלבד, אז מפענח החוקים בונה לו משפט אנגלי מהמילים שזוהו ("אדם עם סכין" → "a person with a knife")
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

**גניבה וונדליזם (UCF-Crime "Anomaly-Videos-Part-4": Shoplifting / Stealing / Vandalism, 200 סרטונים):**
הסרטונים ב-`eval/data/ucf/` (שטוח, בלי תתי-תיקיות). ההערות של UCA מ-GitHub
(`Xuange923/Surveillance-Video-Understanding`, `UCF Annotation/json/UCFCrime_{Train,Val,Test}.json`)
מאוחדות ל-`eval/data/uca_all.json`, ו-`eval/data/ucf_selection_part4.json` = שמות הסרטונים המקומיים.
שלושה תרחישים חדשים, רק כשמבקשים אותם (ברירת המחדל לא משתנה):
`conceal` (הכנסת מוצר לתיק / כיס), `vandal` (ניפוץ / השחתה / גרפיטי), `vehicle` (פריצה לרכב / לקיחת אופנוע).

```bash
python eval/uca_ground_truth.py --sel eval/data/ucf_selection_part4.json --scenarios conceal,vandal,vehicle --out eval/ground_truth_uca_part4.json
.\eval\run_benchmark.ps1 -Gt eval\ground_truth_uca_part4.json
```

על GPU בלי המחשב המקומי: [eval/colab_benchmark.ipynb](eval/colab_benchmark.ipynb) (ה-zip ב-`MyDrive/ucf_bench/`,
אינדוקס בקבוצות של 25, ההתקדמות נשמרת ב-Drive).

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

**אימון שני של הגלאי (`security_v2.pt`, 2026-11-23/24, לא החליף את `security.pt`):** אותו מתכון עם 2,926 תמונות סכין
נוספות (צילומים קרובים מ-Roboflow Universe, `train/pack_exports.py`). ההשוואה, פלט מלא ב-`eval/results/security_v2_2026-11-24.txt`:

```bash
python eval/detector_bench.py eval/ground_truth_uca.json eval/data/ucf --model security_v2.pt --classes gun=weapon,knife=knife,fire=fire,smoke=fire
python eval/offline_eval.py eval/benchmark_v2.db eval/ground_truth_uca.json --extra-raw eval/results/detector_security_v2_raw_2026-11-24.json
```

| | `security.pt` | `security_v2.pt` |
|---|---|---|
| mAP50 סכין, 585 תמונות הסכין החדשות (צילומים קרובים) | 0.000 | 0.952 |
| mAP50 סכין, validation של מצלמות האבטחה (Colab, אותן תמונות) | 0.35 | 0.223 |
| mAP50 gun / fire / smoke, אותן תמונות | 0.69 / 0.73 / 0.84 | 0.685 / 0.729 / 0.834 |
| UCF, הגלאי לבד, סף 0.2: סכין (נמצאו / הוחזרו / חלונות) | 0 / 0 / 46 | 2 / 51 / 46 |
| UCF, הגלאי לבד, סף 0.2: אקדח | 6 / 28 / 81 | 3 / 15 / 81 |
| UCF, חיפוש (CLIP + tracks בסימולציה): F1 כולל | 0.186 | 0.164 |
| UCF, חיפוש: F1 סכין | 0.192 | 0.135 |
| held-out, חיפוש באותה סימולציה (אש + אקדח בלבד): F1 כולל | 0.159 | 0.156 |

מה זה אומר: המודל למד סכין גדולה וברורה (בתמונות האימון הצד הארוך של הסכין הוא 30% מהתמונה בחציון), וב-UCF סכין היא
כמה פיקסלים. שם הוא כמעט רק מוסיף התרעות שווא: מתוך השניות שבהן אמר "סכין", 44 בסרטונים שיש בהם חלון סכין ו-1,118 באחרים;
החיפוש מוצא את אותם 17 חלונות ומחזיר 206 קטעים במקום 131. אקדח, אש ועשן לא השתנו ב-validation; ההבדלים ב-UCF (אקדח 6 → 3
חלונות) קטנים מכדי להסיק מהם. לכן `security.pt` נשאר ברירת המחדל, ו-`security_v2.pt` שמור לבדיקה על סרטונים מבוימים
ב-1080p, שם הסכין גדולה (לא נמדד עדיין). כדי לשפר סכין ב-CCTV צריך תמונות אימון של סכינים קטנות ממצלמות אבטחה.

**סט שלא כוילו עליו (held-out, 2026-11-17):** כל הספים נבחרו על 107 הסרטונים של ה-benchmark, אז המספרים שם אופטימיים.
`eval/uca_ground_truth.py --heldout` בונה סט מ-54 סרטונים אחרים של UCF-Crime (חלק 1: 38 עם אש או אקדח, 16 בלי; 45 חלונות),
והם אונדקסו עם `security.pt` אמיתי (לא סימולציה):

```bash
python eval/uca_ground_truth.py --heldout C:/path/Anomaly-Videos-Part-1 --scenarios fire,weapon --out eval/ground_truth_heldout.json
python eval/index_videos.py C:/path/Anomaly-Videos-Part-1 --db eval/heldout.db --gt eval/ground_truth_heldout.json --only-gt   # בלי שרת
python eval/offline_eval.py eval/heldout.db eval/ground_truth_heldout.json --without gun,knife,fire,smoke                      # בלי ה-tracks של security.pt
```

| held-out | F1 | precision | recall | חלונות שנמצאו |
|---|---|---|---|---|
| CLIP בלבד (בלי ה-tracks של `security.pt`) | 0.149 | 0.10 | 0.29 | 13 מתוך 45 |
| `security.pt` + CLIP | 0.178 | 0.11 | 0.42 | 19 מתוך 45 |

מה זה אומר: (1) הגלאי מוסיף recall אמיתי (אש 9 → 13 חלונות, אקדח 4 → 6) וה-precision לא נפגע; בסרטונים בלי אירוע
הוא מוסיף רק 2 התרעות שווא (60 → 62);
(2) ה-F1 הכולל דומה ל-benchmark (0.18), אבל אש עם CLIP בלבד יורדת מ-0.36 ב-benchmark ל-0.20 כאן (precision 0.36 → 0.15):
הספים של חיפוש הסצנה מתאימים ל-benchmark יותר מאשר לסרטונים חדשים; (3) החולשה היא התרעות שווא של CLIP: "a person with a gun"
מחזיר משהו ב-14 מתוך 16 סרטונים שאין בהם אקדח, ו-"fire" ב-10 מתוך 19. הסט קטן (שני החצאים שלו: 0.21 ו-0.15), ולכן
לא מכיילים עליו: הוא נשאר סט בדיקה.

**מבט שני עם מודל ראייה-שפה (ניסוי, 2026-11-22, לא מחובר לחיפוש):** החולשה היא התרעות שווא, אז `eval/verify_bench.py`
נותן למודל קטן (`SmolVLM2-500M`, מקומי, בלי אימון) שלושה פריימים מכל קטע שהחיפוש החזיר ושאלת כן/לא
("Is a person holding a gun in these images?"), ומשמיט קטעים שה-P(yes) שלהם מתחת לסף.

```bash
python eval/verify_bench.py eval/benchmark_v2.db eval/ground_truth_uca.json --videos eval/data/ucf        # טבלת ספים + AUC
python eval/verify_bench.py eval/heldout.db eval/ground_truth_heldout.json --videos C:/path/Anomaly-Videos-Part-1 --thresholds fire=0.0,weapon=0.1
```

| benchmark (107 סרטונים) | AUC: קטע נכון מול שגוי | F1 בלי אימות | F1 בסף הטוב ביותר |
|---|---|---|---|
| אש | 0.48 | 0.360 | 0.360 (כל סף מוריד, עד 0.295) |
| סכין | 0.53 | 0.192 | 0.204 |
| אקדח | 0.71 | 0.160 | 0.200 (precision 0.11 → 0.53, recall 0.31 → 0.12) |
| חטיפת תיק (11 חלונות) | 0.64 | 0.136 | 0.381 |
| חבילה (7 חלונות, 3 קטעים נכונים) | 0.97 | 0.077 | 0.364 |

AUC 0.5 = ניחוש. באש ובסכין המודל לא מבדיל בין קטע נכון להתרעת שווא (320x240, חלק באינפרא-אדום); באקדח יש אות, במחיר
רוב ה-recall. כשהספים נבחרים על חצי מהסרטונים ונבדקים על החצי השני: 0.277 → 0.306 ו-0.136 → 0.157 (עם ספים שנבחרו על
כל הסרטונים, אופטימי: 0.180 → 0.257). על ה-held-out, עם הספים מה-benchmark (אש בלי סף, אקדח 0.1): 0.178 → 0.219, כולו מאקדח
(0.108 → 0.151; 4 במקום 6 מתוך 14 חלונות, 39 במקום 97 קטעים), וה-AUC שם דומה: אש 0.53, אקדח 0.69.
זמן: 5–15 שניות לקטע על CPU. הפלט המלא: `eval/results/verify_2026-11-22.txt`.
מסקנה: שיפור קטן ולא עקבי במחיר של דקות לשאילתה, ולכן המודל לא חובר ל-`/search`. מודל כזה בלי אימון לא רואה ברזולוציה
הזו יותר ממה ש-CLIP רואה; הצעד הבא שנשאר לבדוק הוא סרטונים ב-1080p.
