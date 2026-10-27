"""Local UCE integration; the upstream editing algorithm stays unchanged."""
from pathlib import Path
import sys

METHOD_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = METHOD_ROOT.parents[1]
sys.path.insert(0, str(WORKSPACE))
