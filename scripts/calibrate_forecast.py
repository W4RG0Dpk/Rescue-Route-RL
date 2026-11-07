"""
Fit the isotonic flood-forecast recalibration used by the recoverability shield.

    python scripts/calibrate_forecast.py      # -> configs/forecast_calibration.json

Training seeds only (300,000+). The agent's observation is not affected.
"""

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rescue_route.env.mission import SCENARIO_PROFILES  # noqa: E402

TRAIN_PROFILES = ["normal"] * 3 + [p for p in SCENARIO_PROFILES if p != "normal"]


def main():
    from rescue_route.forecast.calibration import fit

    stats = fit(range(300_000, 300_420), TRAIN_PROFILES)
    raw = sum(s["brier_raw"] * s["n"] for s in stats.values()) / sum(s["n"] for s in stats.values())
    cal = sum(s["brier_cal"] * s["n"] for s in stats.values()) / sum(s["n"] for s in stats.values())
    for k, s in sorted(stats.items()):
        print(f"lead|alpha {k}: n={s['n']:>7}  mean p {s['mean_p']:.3f} -> observed {s['observed']:.3f}  "
              f"Brier {s['brier_raw']:.4f} -> {s['brier_cal']:.4f}")
    print(f"pooled Brier {raw:.4f} -> {cal:.4f}")


if __name__ == "__main__":
    main()
