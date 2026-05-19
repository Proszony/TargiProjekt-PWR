from __future__ import annotations

import sys
from pathlib import Path

from ui.main_window import run_application


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    return run_application(project_root)


if __name__ == "__main__":
    sys.exit(main())
