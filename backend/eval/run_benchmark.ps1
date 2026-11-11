# Night run: start the backend, index the UCA benchmark videos, score them.
#   cd backend
#   .\eval\run_benchmark.ps1                 # all 107 videos (1-2 h on CPU)
#   .\eval\run_benchmark.ps1 -Limit 5        # smoke test
#   .\eval\run_benchmark.ps1 -Videos "Arson030_x264.mp4,Arson019_x264.mp4" -Force -NoEval   # redo some
# Output: eval\ground_truth_uca_report.json + eval\benchmark.log
param([int]$Limit = 0, [int]$Port = 8011, [string]$Videos = "", [switch]$Force, [switch]$NoEval)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)          # backend/
$env:YOLO_MODEL = "yolov8n.pt,night.pt"                 # not game.pt: it is confidently wrong on CCTV and the probe cannot tell
$env:EXTRA_MODELS = "security.pt"                      # gun / knife / fire / smoke next to the chosen model (not in the 2026-11-07 index)
$env:IMGSZ = "800"
$env:CLIP_RERANK = "1"
# MIN_SCENE / SCENE_Z / MIN_SEMANTIC_UNKNOWN: defaults, tuned on this benchmark (eval\offline_eval.py, eval\sweep_scene.py)
$env:DB_PATH = "eval\benchmark.db"                        # separate index, keeps index.db clean
$env:VIDEO_DIR = "eval\benchmark_videos"
New-Item -ItemType Directory -Force $env:VIDEO_DIR | Out-Null

$server = Start-Process -FilePath ".\venv\Scripts\python.exe" -ArgumentList "-m","uvicorn","main:app","--port","$Port" -PassThru -NoNewWindow -RedirectStandardOutput "eval\server_benchmark.log" -RedirectStandardError "eval\server_benchmark.err.log"
try {
    $up = $false
    for ($i = 0; $i -lt 60 -and -not $up; $i++) {
        Start-Sleep 2
        try { $r = Invoke-RestMethod "http://127.0.0.1:$Port/api"; $up = $r.ok } catch {}
    }
    if (-not $up) { throw "backend did not start on port $Port (see eval\server_benchmark.err.log)" }
    "backend up: models=$($r.models -join ',') clip=$($r.clip)" | Tee-Object -FilePath eval\benchmark.log
    $idxArgs = @("eval\index_videos.py", "eval\data\ucf", "--api", "http://127.0.0.1:$Port")
    if (-not $NoEval) { $idxArgs += @("--gt", "eval\ground_truth_uca.json") }
    if ($Limit -gt 0) { $idxArgs += @("--limit", "$Limit") }
    if ($Videos) { $idxArgs += @("--videos", $Videos) }
    if ($Force) { $idxArgs += "--force" }
    & .\venv\Scripts\python.exe @idxArgs 2>&1 | Tee-Object -FilePath eval\benchmark.log -Append
} finally {
    Stop-Process -Id $server.Id -Force -ErrorAction SilentlyContinue
    Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.Parent.Id -eq $server.Id } | Stop-Process -Force -ErrorAction SilentlyContinue
}
"done; report: eval\ground_truth_uca_report.json"
