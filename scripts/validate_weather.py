"""
Minimal empirical validation of RescueRoute rainfall and wind models (ERA5 / Open-Meteo).

    python scripts/validate_weather.py                       # Mumbai, JJAS monsoon, 2021-2023
    python scripts/validate_weather.py --location chennai --season OND

Outputs:
    reports/weather_validation/weather_validation_report.md
    reports/weather_validation/summary.json
    reports/weather_validation/figures/*.png
    configs/weather_v2_calibrated.json         (Environment V2 weather parameters)
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rescue_route.validation.data import LOCATIONS  # noqa: E402
from rescue_route.validation.pipeline import run_validation  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--location", default="mumbai", choices=sorted(LOCATIONS))
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default="2023-12-31")
    ap.add_argument("--test-year", type=int, default=2023)
    ap.add_argument("--season", default="JJAS", help="JJAS | OND | all")
    ap.add_argument("--p-ref", type=float, default=10.0, help="mm/h mapped to normalised rain intensity 1.0")
    ap.add_argument("--replicates", type=int, default=40)
    ap.add_argument("--refresh", action="store_true", help="re-download instead of using the cache")
    args = ap.parse_args()

    s = run_validation(ROOT, args.location, args.start, args.end, (args.test_year,), args.season,
                       args.p_ref, args.replicates, refresh=args.refresh)
    print("Rain statistics passed:", {m: sum(v["consistent"] for v in t.values()) for m, t in s["rain_comparison"].items()})
    print("Wind statistics passed:", {m: sum(v["consistent"] for v in t.values()) for m, t in s["wind_comparison"].items()})
    print("Report:", ROOT / "reports" / "weather_validation" / "weather_validation_report.md")


if __name__ == "__main__":
    main()
