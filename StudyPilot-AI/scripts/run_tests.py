"""Run the test suite with the current Python interpreter.

Usage: python scripts/run_tests.py [extra pytest args]
Tests never download models or call paid APIs.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    command = [sys.executable, "-m", "pytest", *sys.argv[1:]]
    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
