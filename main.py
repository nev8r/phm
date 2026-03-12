"""
Bearing prediction entrypoint

this file is for running the industrial bearing prediction system

created by zyj

copyright USTC

2026
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from USTC.SSE.BearingPrediction.main import main


if __name__ == "__main__":
    main()
