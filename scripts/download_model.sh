#!/usr/bin/env bash
REPOSITORY="$1"
source /root/vllm_venv/bin/activate
python - "$REPOSITORY" <<'PYTHON'
import sys
from huggingface_hub import snapshot_download

print(snapshot_download(sys.argv[1]))
PYTHON
