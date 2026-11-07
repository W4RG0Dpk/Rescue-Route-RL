import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Original V1 repository, used only as a reference implementation for
# equivalence tests. Tests that need it are skipped when it is absent.
OLD_SRC = ROOT.parent / "old" / "Rescue-Route-main" / "src"
