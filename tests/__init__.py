"""Test package. Puts the repository root and ``src/`` on ``sys.path`` so the
tests run from a plain checkout (``python3 -m unittest discover -s tests -t .``)
without installing anything."""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
