#!/usr/bin/env bash
# Print free disk on the project volume and the size of the biggest project dirs.
# Usage: scripts/check_disk.sh [min_free_gb]   (exits 1 if free space < min_free_gb)
set -euo pipefail
cd "$(dirname "$0")/.."
MIN_GB="${1:-3}"
FREE_GB=$(df -g . | awk 'NR==2 {print $4}')
echo "Free on project volume: ${FREE_GB} GB (threshold ${MIN_GB} GB)"
du -sh .venv .cache data ckpt 2>/dev/null | sort -h || true
if [ "$FREE_GB" -lt "$MIN_GB" ]; then
  echo "LOW DISK: only ${FREE_GB} GB free" >&2
  exit 1
fi
