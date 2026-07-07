"""Enable ``python -m gullivers_router``."""

from __future__ import annotations

import sys

from gullivers_router.cli import main

if __name__ == "__main__":
    sys.exit(main())
