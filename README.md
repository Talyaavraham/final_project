# Video Search Agent

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
| Query understanding | **Claude** (structured output, Hebrew + English, negation, free text) with a rule-based fallback |
| Semantic search | **CLIP** embedding of every track; free text like "holding a phone" is ranked by similarity |
| Multi-object | "person with a dog" = overlap in time **and** proximity in space |
| Identity | tracks that break (occlusion) are merged back into one entity by appearance |
| Outputs | per match: image / clip download; all matches: ZIP with frames, clips, JSON, CSV |
| Long videos | detections are written to SQLite in batches; segments computed once at index time; paginated search |
| Quality | 23 pytest tests (unit + end-to-end), `evaluate.py` reports precision / recall / F1 on labelled ground truth |

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
set ANTHROPIC_API_KEY=sk-ant-...        # Claude query parser (otherwise: rules)
set YOLO_MODEL=yolov8s-worldv2.pt       # open-vocabulary detection
set YOLO_MODEL=game.pt                  # a model fine-tuned on your own classes (backend/train/README.md)
set CLIP_RERANK=0                       # disable CLIP on a weak machine
```

## Tests and evaluation

```bash
cd backend
pytest tests -q
python evaluate.py eval/ground_truth.json      # after labelling a few videos
```

> The hosted page (GitHub Pages) shows the interface with in-browser detection only —
> YOLO, CLIP, Claude and clip export need the backend running locally.
