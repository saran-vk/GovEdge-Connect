#!/usr/bin/env bash
# run_all.sh — Run the complete Week 1 pipeline end-to-end.
# Linux/macOS equivalent of scripts/run_all.ps1
#
# Steps:
#   1. Build ChromaDB index (Track A)
#   2. Synthesize speech corpus (Track B)
#   3. Run WER/CER benchmark (Track B)
#   4. Profile NMT/TTS latency (Track B)
#   5. End-to-end demo
#   6. Generate conclusions report
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Keep HuggingFace cache local to the repo
export HF_HOME="$PROJECT_DIR/.hf_cache"
export TRANSFORMERS_CACHE="$PROJECT_DIR/.hf_cache"

cd "$PROJECT_DIR"

# Activate venv if not already active
if [ -z "${VIRTUAL_ENV:-}" ]; then
    if [ -f "$PROJECT_DIR/.venv/bin/activate" ]; then
        # shellcheck source=/dev/null
        source "$PROJECT_DIR/.venv/bin/activate"
    else
        echo "ERROR: .venv not found. Run scripts/setup_venv.sh first."
        exit 1
    fi
fi

echo "========================================"
echo "  Week 1 Full Pipeline Run"
echo "========================================"

echo ""
echo "[1/6] Building ChromaDB index..."
python -m track_a.build_index

echo ""
echo "[2/6] Synthesizing speech corpus..."
python -m track_b.synth_audio

echo ""
echo "[3/6] Running WER/CER benchmark..."
python -m track_b.benchmark

echo ""
echo "[4/6] Profiling NMT/TTS latency..."
python -m track_b.profiler

echo ""
echo "[5/6] Running end-to-end demo..."
python -m demo.end_to_end --text "Am I eligible for PM Kisan?"

echo ""
echo "[6/6] Generating conclusions report..."
python -m scripts.generate_week1_conclusions

echo ""
echo "========================================"
echo "  Pipeline complete!"
echo "  Reports: docs/week1_conclusions.md"
echo "           track_b/data/results/"
echo "========================================"
