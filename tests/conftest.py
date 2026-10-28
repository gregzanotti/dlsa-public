import os
import sys

import matplotlib

matplotlib.use("Agg")  # no GUI windows during tests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
