# Week 1 one-shot runner: fetch corpus -> build index -> synth audio ->
# WER baseline -> NMT/TTS profiler -> end-to-end demo -> conclusions report.
#
# Usage (from repo root):
#   powershell -ExecutionPolicy Bypass -File scripts/run_all.ps1 [-SkipDemo] [-SkipPdf]
# Optional: set $env:HF_TOKEN before running to unlock gated ai4bharat models.
# All model caches are kept inside the project (.hf_cache) - no global changes.

param(
    [switch]$SkipDemo,
    [switch]$SkipPdf
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Py = Join-Path $Root ".venv\Scripts\python.exe"

if (-not (Test-Path $Py)) {
    Write-Error "virtualenv not found - run scripts/setup_venv.ps1 first"
    exit 1
}

# Keep HuggingFace caches inside the project only.
$env:HF_HOME = Join-Path $Root ".hf_cache"
$env:HUGGINGFACE_HUB_CACHE = Join-Path $env:HF_HOME "hub"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

function Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }

Push-Location $Root
try {
    Step "0/6 Track A - fetch official e-Gov PDFs"
    & $Py -m scripts.fetch_corpus --fetch
    if ($LASTEXITCODE -ne 0) { Write-Host "fetch_corpus skipped (offline?)" }

    Step "1/6 Track A - build ChromaDB index"
    $pdfArg = @()
    if ($SkipPdf) { $pdfArg = @("--skip-pdfs") }
    & $Py -m track_a.build_index @pdfArg
    if ($LASTEXITCODE -ne 0) { throw "build_index failed" }

    Step "2/6 Track B - synthesize en/hi/ta speech corpus (edge-tts)"
    & $Py -m track_b.synth_audio
    if ($LASTEXITCODE -ne 0) { throw "synth_audio failed" }

    Step "3/6 Track B - WER/CER baseline benchmark (faster-whisper)"
    & $Py -m track_b.benchmark
    if ($LASTEXITCODE -ne 0) { throw "benchmark failed" }

    Step "4/6 Track B - NMT/TTS latency profiler"
    & $Py -m track_b.profiler

    if (-not $SkipDemo) {
        Step "5/6 End-to-end demo"
        & $Py -m demo.end_to_end --text "Am I eligible for PM Kisan and which documents are required?"
        if ($LASTEXITCODE -ne 0) { throw "demo failed" }
    }

    Step "6/6 Generate week-1 conclusions"
    & $Py -m scripts.generate_week1_conclusions
    if ($LASTEXITCODE -ne 0) { throw "conclusions generation failed" }

    Write-Host "`n=== Week 1 run complete ===" -ForegroundColor Green
    Write-Host "Reports:"
    Write-Host "  - track_b/data/results/week1_wer_baseline.md"
    Write-Host "  - track_b/data/results/week1_profiler.md"
    Write-Host "  - docs/week1_conclusions.md"
}
finally {
    Pop-Location
}
