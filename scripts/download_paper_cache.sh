#!/bin/bash

# ==============================================================================
# Provenance: [ORIGINAL - MODIFIED]
# Modifications:
# - Added fallback detection for modern 'hf' CLI, virtualenv path (./.venv/bin/hf),
#   and Python direct snapshot_download to handle modern environments seamlessly.
#
# Purpose:
# Downloads the official precomputed LLM response caches from Hugging Face
# (madrylab/platinum-bench-paper-cache) for offline evaluation.
# ==============================================================================

set -e

DATASET_REPO="madrylab/platinum-bench-paper-cache"
LOCAL_DIR="./cache"

# Create destination directory if needed
mkdir -p "$LOCAL_DIR"

# Check if cache files already exist to prevent redundant re-downloads
if find "$LOCAL_DIR" -maxdepth 1 -type f -name "*.pkl" 2>/dev/null | grep -q .; then
    COUNT=$(find "$LOCAL_DIR" -maxdepth 1 -type f -name "*.pkl" | wc -l | tr -d ' ')
    echo "[Info] Cache directory '$LOCAL_DIR' already contains $COUNT .pkl files."
    echo "[Info] Skipping download. To re-download, clear '$LOCAL_DIR'."
    exit 0
fi

echo "[Info] Downloading cache files from '$DATASET_REPO' into '$LOCAL_DIR'..."

if command -v huggingface-cli >/dev/null 2>&1; then
    echo "[Method] Using system 'huggingface-cli'..."
    huggingface-cli download "$DATASET_REPO" \
        --repo-type dataset \
        --include "*.pkl" \
        --local-dir "$LOCAL_DIR"
elif command -v hf >/dev/null 2>&1; then
    echo "[Method] Using system 'hf' CLI..."
    hf download "$DATASET_REPO" \
        --repo-type dataset \
        --include "*.pkl" \
        --local-dir "$LOCAL_DIR"
elif [ -f "./.venv/bin/hf" ]; then
    echo "[Method] Using virtualenv 'hf' CLI (./.venv/bin/hf)..."
    ./.venv/bin/hf download "$DATASET_REPO" \
        --repo-type dataset \
        --include "*.pkl" \
        --local-dir "$LOCAL_DIR"
else
    echo "[Method] Using Python snapshot_download fallback..."
    ./.venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='$DATASET_REPO',
    repo_type='dataset',
    allow_patterns='*.pkl',
    local_dir='$LOCAL_DIR'
)
"
fi

echo "[Success] Cache download finished. Verified downloaded files:"
find "$LOCAL_DIR" -maxdepth 1 -type f -name "*.pkl" -exec ls -lh {} +