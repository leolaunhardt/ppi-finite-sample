"""
Draw every figure of the paper.

    python build_all.py              all of them
    python build_all.py 4            everything in Section 4
    python build_all.py fig_5_4a     one script

Results land in ``output/pdf``; set PPI_FIGURE_DIR to write somewhere else.
Simulation results are cached under ``simdata/``, so the first run of a
figure is slow and every later one is not.
"""

import runpy
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"


def scripts(pattern=None):
    found = sorted(FIGURES.glob("fig_*.py")) + sorted(FIGURES.glob("make_*.py"))
    if pattern:
        found = [p for p in found if pattern in p.stem]
    return found


def main(argv):
    pattern = argv[1] if len(argv) > 1 else None
    if pattern and pattern.isdigit():
        pattern = "fig_%s_" % pattern
    todo = scripts(pattern)
    if not todo:
        print("nothing matches", pattern)
        return 1
    for path in todo:
        print("==", path.name)
        subprocess.run([sys.executable, str(path)], check=True)
    print("\n%d scripts done" % len(todo))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
