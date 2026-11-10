#!/usr/bin/env python
"""Launcher for model.train (keeps the interpreter in isolated mode: python -I scripts/train.py ...)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from model.train import main  # noqa: E402

if __name__ == "__main__":
    main()
