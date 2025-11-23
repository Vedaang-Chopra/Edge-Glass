# code_base/tests/conftest.py
import sys
from pathlib import Path

# This file lives in code_base/tests/
# We want to add code_base/ to sys.path so `from utils...` works everywhere.
CODEBASE_DIR = Path(__file__).resolve().parents[1]
if str(CODEBASE_DIR) not in sys.path:
    sys.path.insert(0, str(CODEBASE_DIR))
