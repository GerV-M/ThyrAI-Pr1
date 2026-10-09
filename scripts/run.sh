
#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:-data}"
OUTPUT_DIR="${2:-results}"

if [[ ! -d "$DATA_DIR" ]]; then
    echo "Error: data directory not found: $DATA_DIR"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/ThyrAI}"

uv run --locked python src/main.py "$DATA_DIR" "$OUTPUT_DIR"
