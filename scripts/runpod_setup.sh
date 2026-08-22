#!/usr/bin/env bash
# RunPod setup script for Formulagate benchmark suite.
# SSH into RunPod on port 22060 and run this:
#
#   bash scripts/runpod_setup.sh
#
# Or use the one-liner:
#   ssh -p 22060 root@<runpod-ip> 'bash -s' < scripts/runpod_setup.sh
#
# This script:
#   1. Installs system packages (Python 3.10+, git)
#   2. Creates a virtual environment
#   3. Installs Formulagate + dependencies
#   4. Downloads MiniLM model (sentence-transformers)
#   5. Fetches 8 benchmark datasets
#   6. Runs the full bench_suite
#   7. Saves results to data/real/bench_suite_report.json
#
# Jupyter is available at http://<runpod-ip>:8888

set -euo pipefail

echo "=== Formulagate RunPod Setup ==="
echo "Started at: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# ─── System packages ─────────────────────────────────────────────────────────
echo "[1/7] Installing system packages ..."
apt-get update -qq
apt-get install -y -qq python3 python3-pip python3-venv git curl wget 2>&1 | tail -1

PYTHON=$(which python3)
echo "  Python: $($PYTHON --version)"

# ─── Clone / sync repository ──────────────────────────────────────────────────
echo "[2/7] Setting up Formulagate repository ..."
REPO_DIR="${REPO_DIR:-/workspace/formulagate}"
if [ -d "$REPO_DIR/.git" ]; then
    echo "  Repository exists, pulling latest ..."
    cd "$REPO_DIR"
    git pull --ff-only
else
    echo "  Cloning repository ..."
    git clone https://github.com/anomalyco/formulagate.git "$REPO_DIR" || cd /workspace && mkdir -p formulagate
fi

cd "$REPO_DIR"

# ─── Virtual environment ──────────────────────────────────────────────────────
echo "[3/7] Creating virtual environment ..."
VENV="$REPO_DIR/.venv"
$PYTHON -m venv "$VENV" --clear
source "$VENV/bin/activate"

echo "  Upgrading pip ..."
pip install --quiet --upgrade pip setuptools wheel

# ─── Install Formulagate + all extras ─────────────────────────────────────────
echo "[4/7] Installing Formulagate with all extras ..."
pip install --quiet -e ".[dev,physics,dense,rag,api]"

# Pre-install benchmarks dependencies
pip install --quiet datasets huggingface_hub requests

echo "  Installed packages:"
pip list 2>/dev/null | grep -E "formulagate|sympy|sentence|faiss|numpy|torch" || true

# ─── Download MiniLM model ────────────────────────────────────────────────────
echo "[5/7] Downloading MiniLM embedding model ..."
python3 -c "
from sentence_transformers import SentenceTransformer
print('  Downloading all-MiniLM-L6-v2 ...')
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print(f'  Model loaded: {model.get_sentence_embedding_dimension()} dims')
" 2>&1 | tail -3

# ─── Fetch benchmark datasets ─────────────────────────────────────────────────
echo "[6/7] Fetching 8 benchmark datasets ..."
python3 scripts/fetch_benchmarks.py --all --limit 100 2>&1 || echo "  (some datasets may need HF token)"

# ─── Run benchmark suite ──────────────────────────────────────────────────────
echo "[7/7] Running benchmark suite ..."
python3 scripts/bench_suite.py --all --folds 3 --runpod 2>&1 | tee data/real/bench_suite.log

echo ""
echo "=== Done ==="
echo "Report: data/real/bench_suite_report.json"
echo "Finished at: $(date -u +%Y-%m-%dT%H:%M:%SZ)"