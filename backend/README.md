# Video Search Agent — Backend (FastAPI + YOLOv8)

## התקנה והרצה

```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows   (במאק/לינוקס: source venv/bin/activate)
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

בהרצה הראשונה ultralytics מוריד אוטומטית את המשקולות `yolov8n.pt` (~6MB).

## איך זה מתחבר לפרונטאנד

1. מריצים את השרת (פורט 8000)
2. פותחים את `video-search-agent.html` בדפדפן
3. מעלים סרטון — הפרונט שולח אותו אוטומטית ל-`POST /index`,
   והשרת מתחיל לאנדקס ברקע (YOLO + tracking, ~2 פריימים לשנייה)
4. כשכותבים שאילתה, הפרונט שולח את השאילתה המפורקת ל-`POST /search`
   ומקבל חזרה מקטעים אמיתיים: זמן, משך, bounding box, צבע, confidence
5. אם השרת לא רץ — הפרונט חוזר אוטומטית למצב דמו (סימולציה)

## Endpoints

| Method | Path      | תיאור |
|--------|-----------|--------|
| POST   | `/index`  | קובץ וידאו (multipart) → אינדוקס ברקע |
| GET    | `/status?video=name.mp4` | מצב האינדוקס: indexing / ready |
| POST   | `/search` | `{video, query:{objects, colors, actions, timeRange}}` → רשימת מקטעים |

## מה יש בפנים

- **זיהוי**: YOLOv8n על מחלקות COCO (person, dog, car, bicycle, motorcycle, truck, bag...)
- **מעקב**: `model.track(persist=True)` — ByteTrack, כך שלכל אובייקט יש track_id רציף
- **צבע**: היסטוגרמת Hue ב-HSV על ה-crop (לאדם — חצי עליון = חולצה)
- **פעולות**: היוריסטיקת מהירות — מרכז ה-bbox זז מהר ⇒ running, לאט ⇒ walking, עומד ⇒ parked
- **שאילתות מרובות אובייקטים** ("איש עם כלב"): חיתוך זמנים בין מקטעים של שתי המחלקות

## שדרוגים מומלצים בהמשך

- החלפת `yolov8n` ב-`yolov8m`/`yolov8l` על GPU לדיוק גבוה יותר
- YOLO-World / Grounding DINO לזיהוי open-vocabulary (כל טקסט חופשי, בלי אימון)
- CLIP re-ranking על ה-crops לשאילתות סמנטיות מורכבות
- החלפת SQLite ב-Postgres + pgvector כשעוברים למולטי-מצלמה
