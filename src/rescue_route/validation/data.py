"""
Historical hourly weather from the Open-Meteo Historical Weather API (ERA5).

Source:   https://archive-api.open-meteo.com/v1/archive  (models=era5)
Data:     ECMWF ERA5 hourly reanalysis (Copernicus Climate Change Service),
          redistributed by Open-Meteo under CC BY 4.0.
Caveat:   ERA5 is a ~0.25 deg (~28 km) gridded reanalysis, not a point
          anemometer / rain-gauge record. It smooths convective rainfall
          peaks and local wind extremes.

Downloaded data are cached as CSV in ``data/weather`` together with a
metadata JSON, so validation is reproducible offline.

Direction conventions
---------------------
ERA5 / Open-Meteo ``wind_direction_10m`` is METEOROLOGICAL: the direction
the wind blows FROM, clockwise from north. RescueRoute's ``WindModel`` uses
the direction the air moves TOWARD (0 = north, 90 = east). Hence

    theta_sim = (theta_met + 180) mod 360

Vector components (east = +x, north = +y):

    u = -S sin(theta_met),   v = -S cos(theta_met)
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

API_URL = "https://archive-api.open-meteo.com/v1/archive"
VARIABLES = ["precipitation", "rain", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m"]


@dataclass(frozen=True)
class Location:
    key: str
    name: str
    latitude: float
    longitude: float
    note: str


LOCATIONS = {
    "mumbai": Location("mumbai", "Mumbai, Maharashtra", 19.0760, 72.8777,
                       "Coastal megacity with recurrent monsoon urban flooding (e.g. 26 July 2005)."),
    "chennai": Location("chennai", "Chennai, Tamil Nadu", 13.0827, 80.2707,
                        "North-east monsoon flooding (e.g. December 2015)."),
    "kolkata": Location("kolkata", "Kolkata, West Bengal", 22.5726, 88.3639,
                        "Cyclone- and monsoon-driven waterlogging."),
    "guwahati": Location("guwahati", "Guwahati, Assam", 26.1445, 91.7362,
                         "Brahmaputra basin, frequent monsoon floods."),
}


def fetch_era5(
    location: Location,
    start: str,
    end: str,
    cache_dir: Path,
    refresh: bool = False,
) -> pd.DataFrame:
    """Download (or load cached) hourly ERA5 data and add derived columns."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = f"era5_{location.key}_{start}_{end}"
    csv_path = cache_dir / f"{stem}.csv"
    meta_path = cache_dir / f"{stem}.meta.json"

    if csv_path.exists() and not refresh:
        df = pd.read_csv(csv_path, parse_dates=["time"])
    else:
        params = {
            "latitude": location.latitude,
            "longitude": location.longitude,
            "start_date": start,
            "end_date": end,
            "hourly": ",".join(VARIABLES),
            "models": "era5",
            "wind_speed_unit": "ms",
            "precipitation_unit": "mm",
            "timezone": "GMT",
        }
        last_err: Optional[Exception] = None
        for attempt in range(4):
            try:
                resp = requests.get(API_URL, params=params, timeout=90)
                resp.raise_for_status()
                payload = resp.json()
                break
            except Exception as err:  # network hiccup: retry with backoff
                last_err = err
                time.sleep(2 ** attempt)
        else:
            raise RuntimeError(f"Open-Meteo download failed: {last_err}")

        hourly = payload["hourly"]
        df = pd.DataFrame({"time": pd.to_datetime(hourly["time"])})
        for v in VARIABLES:
            df[v] = pd.to_numeric(pd.Series(hourly[v]), errors="coerce")
        df.to_csv(csv_path, index=False)
        meta = {
            "source": "Open-Meteo Historical Weather API",
            "url": API_URL,
            "model": "ERA5 (ECMWF / Copernicus C3S reanalysis)",
            "license": "CC BY 4.0 (Open-Meteo); ERA5 under Copernicus licence - attribution required",
            "location": asdict(location),
            "grid_cell_returned": {"latitude": payload.get("latitude"), "longitude": payload.get("longitude"),
                                   "elevation_m": payload.get("elevation")},
            "start": start,
            "end": end,
            "timezone": "GMT",
            "units": payload.get("hourly_units"),
            "variables": VARIABLES,
            "downloaded_utc": pd.Timestamp.utcnow().isoformat(),
            "rows": int(len(df)),
        }
        meta_path.write_text(json.dumps(meta, indent=2))

    return add_derived_columns(df)


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Forward-fill isolated gaps (ERA5 is complete; this only guards API glitches).
    missing = df[VARIABLES].isna().sum().sum()
    if missing:
        df[VARIABLES] = df[VARIABLES].interpolate(limit=3).bfill().ffill()
    df.attrs["n_missing_filled"] = int(missing)

    met = np.radians(df["wind_direction_10m"].to_numpy())
    speed = df["wind_speed_10m"].to_numpy()
    df["u"] = -speed * np.sin(met)
    df["v"] = -speed * np.cos(met)
    df["direction_toward_deg"] = (df["wind_direction_10m"] + 180.0) % 360.0
    df["gust_excess"] = np.maximum(0.0, df["wind_gusts_10m"] - df["wind_speed_10m"])
    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month
    return df


def select_season(df: pd.DataFrame, season: str) -> pd.DataFrame:
    """'JJAS' = south-west monsoon (June-September); 'all' = whole year."""
    if season.lower() == "all":
        return df
    if season.upper() == "JJAS":
        return df[df["month"].isin([6, 7, 8, 9])]
    if season.upper() == "OND":
        return df[df["month"].isin([10, 11, 12])]
    raise ValueError(f"Unknown season '{season}'.")


def contiguous_segments(df: pd.DataFrame) -> list:
    """Split into runs of consecutive hours (seasonal subsets have gaps between years)."""
    t = df["time"].to_numpy()
    if len(t) == 0:
        return []
    breaks = np.where(np.diff(t).astype("timedelta64[h]").astype(int) != 1)[0] + 1
    idx = np.split(np.arange(len(df)), breaks)
    return [df.iloc[i] for i in idx if len(i) > 1]
