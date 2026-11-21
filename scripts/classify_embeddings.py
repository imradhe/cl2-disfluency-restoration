#!/usr/bin/env python3
"""Full-corpus frozen versus task-trained text embeddings."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from disfluency.embedding_classification import main

if __name__ == '__main__':
    main()
