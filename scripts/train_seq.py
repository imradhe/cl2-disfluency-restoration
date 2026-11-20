#!/usr/bin/env python
"""Launcher for model.train_seq (keeps the interpreter in isolated mode: python -I scripts/train_seq.py ...)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from model.train_seq import main  # noqa: E402

if __name__ == "__main__":
    main()
