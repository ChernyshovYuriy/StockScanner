"""
conftest.py — adds the project root to sys.path so that test modules in the
tests/ sub-directory can import top-level project modules directly.
"""

import sys
from pathlib import Path

# Insert project root (the directory containing this file) at the front of the
# path so that 'import migrate_report', 'import report_html', etc. resolve
# correctly regardless of which directory pytest is invoked from.
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
