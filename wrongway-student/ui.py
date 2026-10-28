#!/usr/bin/env python3
import sys
from pathlib import Path

# Chuyển tiếp tới wrongway/ui.py
wrongway_dir = Path(__file__).resolve().parent / "wrongway"
sys.path.insert(0, str(wrongway_dir))

if __name__ == "__main__":
    from ui import main
    main()

