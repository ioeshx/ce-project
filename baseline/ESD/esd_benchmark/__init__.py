"""Local ESD integration using the upstream teacher/student objective."""
from pathlib import Path
import sys

METHOD_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = METHOD_ROOT.parents[1]
sys.path.insert(0, str(WORKSPACE))
