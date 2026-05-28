"""
Shared test fixtures. Adds the project root to sys.path so tests can
import the top-level modules (bandits, log_store, etc.) without an
explicit install step.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
