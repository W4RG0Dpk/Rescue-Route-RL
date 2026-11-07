"""The vectorised V1.1 simulator must reproduce the original V1 code exactly."""

import importlib
import sys

import numpy as np
import pytest

from conftest import OLD_SRC

pytestmark = pytest.mark.skipif(not OLD_SRC.exists(), reason="original V1 repository not available")


def _import_old():
    """Import the original package under the alias 'old_rescue_route'."""
    if "old_rescue_route" in sys.modules:
        return sys.modules["old_rescue_route"]
    spec = importlib.util.spec_from_file_location(
        "old_rescue_route", OLD_SRC / "rescue_route" / "__init__.py", submodule_search_locations=[str(OLD_SRC / "rescue_route")]
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["old_rescue_route"] = mod
    spec.loader.exec_module(mod)
    return mod


def _old_env_cls():
    _import_old()
    return importlib.import_module("old_rescue_route.env.rescue_route_env").RescueRouteEnv


@pytest.mark.parametrize("seed", [0, 7, 123, 2024])
def test_full_episode_equivalence(seed):
    from rescue_route.env import RescueRouteEnv

    old = _old_env_cls()()
    new = RescueRouteEnv(legacy_wind_obs=True, scenario_version="v1")
    o_obs, _ = old.reset(seed=seed)
    n_obs, _ = new.reset(seed=seed)
    rng = np.random.default_rng(seed)
    for _ in range(200):
        a = int(rng.integers(0, 9))
        o = old.step(a)
        n = new.step(a)
        assert o[1] == pytest.approx(n[1], abs=1e-9)
        assert o[2] == n[2]
        np.testing.assert_array_equal(old.flood_model.flood_grid, new.flood_model.flood_grid)
        np.testing.assert_allclose(o[0]["vector_state"], n[0]["vector_state"], atol=1e-6)
        np.testing.assert_allclose(o[0]["spatial_tensor"], n[0]["spatial_tensor"], atol=1e-5)
        assert o[4]["terminated_reason"] == n[4]["terminated_reason"]
        if o[2]:
            break


def test_flood_and_forecast_long_rollout_equivalence():
    """Flood + forecast with no UAV in the loop, 60 steps, heavy rain."""
    _import_old()
    OldFlood = importlib.import_module("old_rescue_route.hazards.flood").DynamicFloodModel
    OldForecast = importlib.import_module("old_rescue_route.forecast.forecast_model").ForecastModel
    from rescue_route.forecast import ForecastModel
    from rescue_route.hazards import DynamicFloodModel
    from rescue_route.map import load_map
    from rescue_route.weather import WindModel

    grid = load_map("manhattan32").grid_map
    kw = dict(initial_flood_ratio=0.03, expansion_rate=0.06, rain_intensity=0.9, seed=5)
    a, b = OldFlood(32, 32, **kw), DynamicFloodModel(32, 32, **kw)
    a.reset(grid, safe_coords=[(1, 1)], seed=5)
    b.reset(grid, safe_coords=[(1, 1)], seed=5)
    wind = WindModel(base_speed=11, seed=3)
    wind.reset(seed=3)
    fa, fb = OldForecast(horizon_steps=6, alpha_fc=0.7), ForecastModel(horizon_steps=6, alpha_fc=0.7)
    for _ in range(60):
        ws = wind.step()
        a.step(grid, ws)
        b.step(grid, ws)
        np.testing.assert_array_equal(a.flood_grid, b.flood_grid)
        assert a.rain_intensity == b.rain_intensity
    sa = fa.generate_forecast(a.flood_grid, grid, ws, a.rain_intensity)
    sb = fb.generate_forecast(b.flood_grid, grid, ws, b.rain_intensity)
    np.testing.assert_allclose(sa.flood_prob_sequence, sb.flood_prob_sequence, atol=1e-5)
