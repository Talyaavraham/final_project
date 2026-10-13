# Video Search Agent

Search inside a video using natural-language queries (English / Hebrew),
e.g. "find a person wearing a red shirt" or "מצא איש שרץ עם כלב".
The system returns the matching timestamps, clips and bounding boxes.

Project by Maya Lax & Talya Avraham — course 67546.

## Structure

- `index.html` — the web interface (chat search, timeline, results panel).
  Works on its own in demo mode; connects to the backend automatically when it runs.
- `backend/` — FastAPI + YOLOv8 server (detection, tracking, SQLite index, search).
  See `backend/README.md` for setup.

## Run

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Then open `index.html` in the browser and upload a video (MP4 / H.264).

> The hosted page (GitHub Pages) shows the interface in demo mode only —
> real detections need the backend running locally.
