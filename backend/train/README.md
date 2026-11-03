# אימון הגלאי על אובייקטים חדשים (המשחק ב-ex4_demo)

YOLOv8 המוכן מזהה רק את 80 מחלקות COCO, ו-YOLO-World לא מזהה כלום בפיקסל-ארט
(נבדק: גם עם סף 0.05 ו-imgsz 1280, עם מילון כמו "character / tree / apple" — אפס תיבות).
לכן מאמנים מודל קטן על הפריימים של המשחק עצמו.

## 1. תיוג אוטומטי — `autolabel.py`

המשחק מצייר כל אובייקט בצבע אחיד, אז במקום לתייג ידנית ב-CVAT, הסקריפט מפלח
לפי צבע והופך כל כתם לתיבה:

| מחלקה | איך מזהים |
|---|---|
| `character` | הדבר היחיד עם קו-מתאר שחור (חוץ מטקסט ה-HUD בפינה, שמוחרג) |
| `tree` | גזע חום כהה + הצמרת הירוקה שמעליו (נחתך בקו הקרקע) |
| `fruit` | עיגולים קטנים אדומים / צהובים. הצבע (אדום/צהוב) מחושב ממילא באינדקסר |
| `sun` | עיגול צהוב גדול |

יום/לילה במשחק מכהה את כל הפיקסלים באותו יחס, אז כל פריים מנורמל לפי צבע השמיים
לפני הפילוח. פריימים חשוכים מדי מדולגים.

```bash
cd backend
python train/autolabel.py videos/ex4_demo_2025_26_winter.mp4 train/dataset --every 8 --debug
```

יוצר `train/dataset/{images,labels}/{train,val}` ו-`data.yaml`; עם `--debug` גם
`train/dataset/debug/*.jpg` עם התיבות מצוירות — כדאי לפתוח כמה ולוודא.
על הסרטון הזה: 223 פריימים, 220 דמויות, ~1200 עצים, ~2300 פירות, 133 שמשות.

## 2. אימון

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').train(data='train/dataset/data.yaml', epochs=10, imgsz=800, batch=8, workers=0, device='cpu', project='train/runs', name='game', close_mosaic=3)"
```

(או `yolo detect train model=yolov8n.pt data=train/dataset/data.yaml epochs=10 imgsz=800`.)
`imgsz=800` ולא 640 כי הפירות הם ~24 פיקסלים ב-1080p. על CPU זה בערך שעה (ב-960 ו-25 epochs: כ-3 שעות);
ב-Google Colab עם GPU כמה דקות (להעלות את `train/dataset` ולהריץ את אותה פקודה).
התוצאה: `train/runs/game/weights/best.pt`, והמדדים (mAP לכל מחלקה) ב-`results.csv`.

### ב-Google Colab (מומלץ, כמה דקות על GPU)

1. `python train/autolabel.py ...` כמו למעלה, ואז לכווץ את התיקייה: `train/dataset.zip` (כ-25MB).
2. לפתוח את [colab_train.ipynb](colab_train.ipynb) ב-https://colab.research.google.com (File → Upload notebook),
   לבחור Runtime → T4 GPU, ולהריץ את התאים לפי הסדר. תא 2 מבקש להעלות את ה-zip.
3. התא האחרון מוריד את `best.pt`. לשמור אותו כ-`backend/game.pt`.

המחברת מאמנת 40 epochs ב-imgsz 800 ומדפיסה mAP לכל מחלקה על ה-validation.

## 3. חיבור למערכת

```bash
copy train\runs\game\weights\best.pt game.pt
set YOLO_MODEL=game.pt
set IMGSZ=800
uvicorn main:app --port 8000
```

האינדקסר לוקח את שמות המחלקות מקובץ המשקולות, אז לא צריך לשנות קוד.
עדיף: `set YOLO_MODEL=yolov8n.pt,game.pt`, ואז כל סרטון מקבל אוטומטית את המודל שמתאים לו
(סרטון מהמציאות → COCO, המשחק → `game.pt`) לפי בדיקה של 20 פריימים. מה שכן
חובר:

- `query_parser.VOCAB_OBJECTS` — מילים בעברית/אנגלית למחלקות החדשות
  (דמות/שחקן, עץ, פרי/תפוח/לימון, שמש).
- `/parse` מעביר למפענח רק את המחלקות שנמצאו בסרטון, ו-"איש" הופך ל-`character`
  כשבסרטון אין `person` (`CLASS_FALLBACK`).
- צבע הפרי מגיע מהאינדקסר, אז "תפוח אדום" = `fruit` + `red`, "פרי צהוב" = `fruit` + `yellow`.

## תוצאות (אומן ב-Colab על T4, 40 epochs, 2026-10-21)

Validation של 36 פריימים, 589 תיבות:

| מחלקה | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| character | 0.998 | 1.000 | 0.995 | 0.950 |
| tree | 0.981 | 0.978 | 0.994 | 0.987 |
| fruit | 1.000 | 0.978 | 0.995 | 0.946 |
| sun | 0.944 | 0.881 | 0.906 | 0.833 |
| **הכול** | 0.981 | 0.959 | **0.973** | 0.929 |

אינדוקס הסרטון המלא (60 שניות, 3 פריימים/שנייה, ללא CLIP) לוקח כ-30 שניות על CPU
ומוצא 1 דמות (track אחד לאורך כל הסרטון), 108 עצים, 173 פירות (86 אדומים, 87 צהובים), 6 שמשות.
שאילתות שעובדות: "פרי צהוב", "שמש", "דמות ליד עץ עם תפוח אדום", "yellow fruit between 0:10 and 0:20".

מגבלות ידועות:

- **פעולות של הדמות**: המצלמה במשחק עוקבת אחרי הדמות, אז על המסך היא כמעט לא זזה
  (העולם זז). לכן "דמות רצה" מחזיר 0 — זיהוי הפעולות מניח מצלמה סטטית.
- **צבע העצים**: 43 מתוך 108 העצים נקראים "blue" במקום "green", כי תיבת העץ כוללת
  הרבה שמיים בין העלים והצבע נקבע בהצבעת פיקסלים (`indexer.dominant_color`).
  "עץ" בלי צבע מחזיר את כולם; הפירות והשמש מלאים ונשארים אדום/צהוב.
- התיוג האוטומטי מצייר תיבה לכל גזע, אז צמרות חופפות של עצים צמודים מקבלות
  תיבות חופפות.

## גלאי לילה (`night.pt`) — `night_dataset.py`

בסרטון מצלמת אבטחה חשוך (640×352, אינפרא, פנס מסנוור) YOLOv8n לא מצא אף אדם,
גם עם סף 0.15 ו-yolov8s: המודל מוציא "person" בביטחון 0.15–0.3 פעם בכמה שניות,
בלי מזהה מעקב, ו-`search.make_segment` זורק גילוי בודד חלש כרעש (נכון, אחרת
הסרטון מתמלא ב-vase ו-train). הפתרון: fine-tuning על תמונות לילה.

1. **ExDark** (Exclusively Dark Image Dataset): 7,363 תמונות בתאורה נמוכה עם תיבות
   ל-12 מחלקות (People, Car, Bus, Bicycle, Motorbike, Dog, Cat, Boat, Bottle, Chair,
   Cup, Table). יורד אוטומטית מ-Hugging Face (`dronefreak/ExDark`, פורמט YOLO, 640×640,
   ~280 MB). שמות המחלקות ממופים לשמות COCO (People→person, Motorbike→motorcycle…),
   אז המפענח, הטבלאות ב-`indexer` וה-UI לא משתנים.
2. **פריימים משלכן** (אופציונלי אבל מומלץ, ExDark צולם מקרוב ובאיכות טובה):
   ```bash
   python train/night_dataset.py frames videos/dark.mp4 --every 1.5
   ```
   מייצר `train/night/frames/dark_000042.0.jpg …`. מעלים ל-Roboflow (או CVAT), מתייגים
   person / car, מייצאים בפורמט "YOLOv8" ומכווצים ל-zip. הסקריפט מכפיל כל פריים כזה ×4
   כדי שלא ייבלע ב-6,000 התמונות של ExDark.
3. **אימון ב-Colab**: [colab_train_night.ipynb](colab_train_night.ipynb) — מוריד את ExDark,
   מקבל את ה-zip (אפשר לדלג), בונה את הדאטהסט, מאמן `yolov8s` ב-imgsz 800, 25 epochs
   (כשעה על T4; `epochs=12` או `yolov8n.pt` למהיר יותר) ומוריד `best.pt`.
   כדאי טוקן חינמי של Hugging Face ב-Secrets של קולאב (`HF_TOKEN`), אחרת ההורדה
   נחסמת זמנית (429) והסקריפט ממתין ומנסה שוב.
4. **חיבור**: `copy best.pt backend\night.pt` ואז
   `YOLO_MODEL=yolov8n.pt,game.pt,night.pt` + `IMGSZ=800`. סרטון חשוך יבחר ב-`night.pt`
   לבד (בדיקת 20 הפריימים).

בנייה מקומית (לבדיקה, בלי Colab): `python train/night_dataset.py build --own DIR`
(`--no-exdark` לפריימים שלכן בלבד, `--classes person,car --limit 200` לניסוי מהיר).
הבדיקות ב-`tests/test_night_dataset.py` מכסות את המיפוי והמיזוג בלי הורדה.

ציפייה: אנשים בלילה בגודל סביר יזוהו הרבה יותר טוב; אנשים של ~20 פיקסלים בסרטון
מסונוור יישארו קשים. לדמו עדיף גם סרטון לילה ב-720p ומעלה.

## אותו רעיון לסרטון אחר

אם המשחק שלכן, הכי מדויק לייצר את התיבות מהקוד של המשחק (הוא יודע איפה כל
אובייקט מצויר). אם זה סרטון מהעולם האמיתי, `autolabel.py` לא רלוונטי — מתייגים
150–300 פריימים ב-CVAT / Roboflow ומייצאים בפורמט YOLO לאותו מבנה תיקיות.
