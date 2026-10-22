"""Project tests."""
"""Make the flat runtime source tree available to unittest discovery."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
