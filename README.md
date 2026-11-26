# Video Search Agent

<div dir="rtl">

> מה השתנה ומתי: [CHANGELOG.md](CHANGELOG.md)

## שאלת המחקר

**האם אפשר לזהות, בזמן אמת ובשידור חי, פעולה מסוכנת שהמשתמשת מגדירה בשפה חופשית (למשל "מישהו מרים סכין"), עם מעט התרעות שווא, על ידי ניתוח של תנועת השלד ומיקום האובייקטים לאורך זמן? והאם מודל שלומד את רצפי השלד והאובייקטים עושה זאת טוב יותר מכללים שנכתבו ביד?**

### הבעיה
גלאי אובייקטים יודעים לזהות **עצמים** ("יש סכין"), אבל משתמשים רוצים התרעה על **פעולות** ("מישהו מרים סכין"). פעולה מתרחשת לאורך זמן ובין אדם לאובייקט. לכן סכין שמונחת על שולחן, או סכין שחותכים בה ירקות, לא אמורות להפעיל התרעה. התרעה על כל סכין בפריים יוצרת הרבה התרעות שווא, ובמערכת התרעה חיה זה הופך אותה ללא שמישה.

### שאלות משנה
1. **משפה לכלל:** באיזו מידה אפשר לתרגם בקשה חופשית בעברית או באנגלית לכלל שהמערכת מריצה? אילו בקשות היא מבינה, ואילו היא מסרבת לקבל במקום ליצור כלל שגוי?
2. **זיהוי הפעולה:** בהשוואה בין שלוש גישות:
   - **(א)** אובייקט בלבד: התרעה כשרואים סכין.
   - **(ב)** כלל שנכתב ביד על שלד ואובייקט: היד שאחזה בסכין עולה מעל הכתף.
   - **(ג)** מודל נלמד על רצפים של נקודות שלד ומסגרות אובייקטים.

   איזו גישה מגיעה ל-recall הגבוה ביותר, עם הכי מעט התרעות שווא לשעה ועם השהיה קצרה בין תחילת הפעולה להתרעה?
3. **עמידות:** איך הגישות מתמודדות עם פעולות מבלבלות (חיתוך, הרמת טלפון, הנפת יד ריקה), עם גלאי שמאבד את הסכין בטשטוש תנועה, ועם זוויות ותאורה שונות?
4. **זמן אמת:** כמה פריימים לשנייה וכמה השהיה מתקבלים על GPU ועל CPU, והאם זה מספיק להתרעה בזמן?

### ההשערה
שילוב של **תנוחת הגוף + האובייקט + הזמן** (גישות ב' ו-ג') יוריד את התרעות השווא באופן משמעותי לעומת זיהוי האובייקט בלבד (גישה א'), כמעט בלי לפגוע ב-recall. מודל נלמד (ג') יתמודד עם פעולות מבלבלות טוב יותר מכלל קבוע (ב').

### איך נמדוד
- **דאטה:** סרטונים שנצלם בעצמנו: הרמות סכין אמיתיות (עם סכין צעצוע או אביזר), לצד פעולות מבלבלות וקטעים רגילים ללא אירוע. כל אירוע מתויג עם זמן התחלה וזמן סיום. החלוקה ל-train / validation / test תהיה לפי סרטונים, וגם לפי אנשים כשאפשר.
- **מדדים:** recall של אירועים, **התרעות שווא לשעה**, השהיה מתחילת הפעולה עד ההתרעה, ו-fps.
- **הבנת השפה:** סט של ניסוחים, בעברית ובאנגלית, לכל סוג כלל, כולל בקשות שהמערכת לא תומכת בהן. נמדוד כמה מהם תורגמו נכון.

### מה מחוץ לתחום
זיהוי פנים, אודיו, ומעקב בין כמה מצלמות.

</div>

## Research Question

**Can a dangerous action that the user defines in free language (e.g. "someone raises a knife") be detected in real time on a live camera, with few false alarms, by reasoning over body pose and object positions over time? And does a model that learns from pose-and-object sequences do this better than hand-written rules?**

### The problem
Object detectors find **things** ("there is a knife"), but people want alerts about **actions** ("someone raises a knife"). An action unfolds over time and between a person and an object. A knife lying on a table, or one used to cut vegetables, should not raise an alert. Alerting on every knife in the frame produces so many false alarms that a live alert system becomes unusable.

### Sub-questions
1. **Language to rule:** how well can a free request in Hebrew or English be turned into a rule the system can run? Which requests does it understand, and which does it refuse instead of building a wrong rule?
2. **Detecting the action:** we compare three approaches:
   - **(a)** object only: alert when a knife is seen.
   - **(b)** a hand-written rule on pose and object: the wrist that held the knife rises above the shoulder.
   - **(c)** a model learned on sequences of keypoints and object boxes.

   Which one reaches the highest recall, with the fewest false alarms per hour and the shortest delay from the start of the action to the alert?
3. **Robustness:** how do the approaches handle confusing actions (cutting, raising a phone, waving an empty hand), a detector that loses the knife in motion blur, and different angles and lighting?
4. **Real time:** what frame rate and latency do we get on GPU and on CPU, and is that fast enough to alert in time?

### Hypothesis
Combining **body pose + object + time** (approaches b and c) will cut false alarms substantially compared with object detection alone (a), with little loss of recall. The learned model (c) will handle confusing actions better than a fixed rule (b).

### How we measure
- **Data:** videos we record ourselves: real knife raises (with a toy or prop knife), confusing actions, and normal footage with no event. Every event is labelled with a start and end time. The train / validation / test split is by video, and by person where possible.
- **Metrics:** event recall, **false alarms per hour**, delay from the start of the action to the alert, and fps.
- **Language understanding:** a set of phrasings, in Hebrew and English, for each rule kind, including requests the system does not support. We measure how many are turned into the right rule.

### Out of scope
Face recognition, audio, and tracking across several cameras.

---

Search inside a long video using natural-language queries (English / Hebrew),
e.g. "find a person wearing a red shirt", "a woman pushing a stroller" or
"מצא איש שרץ עם כלב". The system returns the matching timestamps with tracked
bounding boxes, and exports them as **annotated frames (JPEG), clips (MP4)
and a ZIP with results.json / results.csv**.

Project by Maya Lax & Talya Avraham — course 67546.

## What it does

| Step | How |
|---|---|
| Detection + tracking | YOLOv8 (all 80 COCO classes) or **YOLO-World** (open vocabulary) + ByteTrack, ~3 frames/s |
| Attributes | colour of the shirt / object, speed → running / walking / parked, entering / exiting the frame |
| Query understanding | local, no outside API: rule-based vocabulary (Hebrew + English) for search; **our own trained model** (`backend/nlu/`) for live alert requests, rules as fallback |
| Semantic search | **CLIP** embedding of every track; free text like "holding a phone" is ranked by similarity. Whole-frame CLIP every second finds scenes with no object class ("fire") |
| Multi-object | "person with a dog" = overlap in time **and** proximity in space |
| Identity | tracks that break (occlusion) are merged back into one entity by appearance |
| Outputs | per match: image / clip download; all matches: ZIP with frames, clips, JSON, CSV |
| Long videos | detections are written to SQLite in batches; segments computed once at index time; paginated search |
| Quality | 40 pytest tests (unit + end-to-end), `evaluate.py` reports precision / recall / F1 per scenario; `backend/eval/` benchmarks the security scenarios on UCF-Crime (UCA annotations) |

## Structure

- `index.html` — the web interface (chat search, timeline, results, downloads).
  Works on its own with in-browser detection (TensorFlow.js COCO-SSD); connects to the backend automatically when it runs.
- `backend/` — FastAPI server: indexing, search, parsing, export. See `backend/README.md`.

## Run

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Then open **http://localhost:8000** in the browser and upload a video (MP4 / H.264).
(`index.html` also works when opened straight from disk; it looks for the backend on port 8000.)

Optional:

```bash
set YOLO_MODEL=yolov8s-worldv2.pt       # open-vocabulary detection
set YOLO_MODEL=game.pt                  # a model fine-tuned on your own classes (backend/train/README.md)
set YOLO_MODEL=yolov8n.pt,game.pt       # several models: the best one is picked per video
set YOLO_MODEL=yolov8n.pt,night.pt      # night.pt: fine-tuned on low-light footage (backend/train/night_dataset.py)
set EXTRA_MODELS=security.pt            # gun / knife / fire / smoke detector run next to the chosen model
set CLIP_RERANK=0                       # disable CLIP on a weak machine
```

## Tests and evaluation

```bash
cd backend
pytest tests -q
python evaluate.py eval/ground_truth.json      # after labelling a few videos
```

> The hosted page (GitHub Pages) shows the interface with in-browser detection only —
> YOLO, CLIP and clip export need the backend running locally.
