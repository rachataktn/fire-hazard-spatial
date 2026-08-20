#!/usr/bin/env python3
from experiment_core import main

if __name__ == "__main__":
    import sys
    sys.argv.insert(1, "evaluate")
    sys.argv.extend(["--task", "segment"])
    raise SystemExit(main())
