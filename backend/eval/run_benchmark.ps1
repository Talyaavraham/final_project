# Night run: start the backend, index the UCA benchmark videos, score them.
#   cd backend
#   .\eval\run_benchmark.ps1                 # all 107 videos (1-2 h on CPU)
#   .\eval\run_benchmark.ps1 -Limit 5        # smoke test
# Output: eval\ground_truth_uca_report.json + eval\benchmark.log
param([int]$Limit = 0, [int]$Port = 8011)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)          # backend/
$env:YOLO_MODEL = "yolov8n.pt,game.pt,night.pt"
$env:IMGSZ = "800"
$env:CLIP_RERANK = "1"
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
    $args = @("eval\index_videos.py", "eval\data\ucf", "--api", "http://127.0.0.1:$Port", "--gt", "eval\ground_truth_uca.json")
    if ($Limit -gt 0) { $args += @("--limit", "$Limit") }
    & .\venv\Scripts\python.exe @args 2>&1 | Tee-Object -FilePath eval\benchmark.log -Append
} finally {
    Stop-Process -Id $server.Id -Force -ErrorAction SilentlyContinue
    Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.Parent.Id -eq $server.Id } | Stop-Process -Force -ErrorAction SilentlyContinue
}
"done; report: eval\ground_truth_uca_report.json"
