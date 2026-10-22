"""Portable entry point; all paths are relative to this project directory."""
from pathlib import Path
import os
import sys


def main() -> None:
    if sys.version_info < (3, 13):
        raise SystemExit(
            "Traj-LTLf requires Python 3.13+. On Windows: py -3.13 -m venv .venv; "
            ".\\.venv\\Scripts\\Activate.ps1; python -m pip install -r requirements.txt"
        )
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root / "src"))
    os.chdir(root)
    from cli import main as run
    run()


if __name__ == "__main__":
    main()
