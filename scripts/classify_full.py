#!/usr/bin/env python3
"""Run binary and per-type classification on all processed Switchboard data."""
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from disfluency.classification import main

if __name__ == '__main__':
    main()
