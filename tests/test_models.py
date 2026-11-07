"""Mathematical property tests for the stochastic models (report section 11.1)."""

import numpy as np
import pytest

from rescue_route.forecast import ForecastModel
from rescue_route.hazards import DynamicFloodModel
from rescue_route.map import load_map
from rescue_route.payload import PayloadStabilityModel
from rescue_route.weather import RainModelV1, WindModel, WindState


def wind_state(speed=8.0, direction=90.0, gust=0.0):
    m = WindModel(base_speed=speed, speed_std=0.0)
    m.current_speed, m.current_direction, m.current_gust = speed, direction, gust
    return m.get_state(), m


# ----------------------------------------------------------------------------- rain
def test_rain_bounds_and_no_nan():
    x = RainModelV1.simulate(5000, mean=0.6, variability=0.4, correlation=0.9, seed=1)
    assert np.all(np.isfinite(x)) and x.min() >= 0.0 and x.max() <= 1.0


def test_rain_persistence_matches_rho_without_clipping():
    x = RainModelV1.simulate(40000, mean=0.5, variability=0.05, correlation=0.8, seed=2)
    xc = x - x.mean()
    rho = np.dot(xc[:-1], xc[1:]) / np.dot(xc, xc)
    assert rho == pytest.approx(0.8, abs=0.02)
    assert x.std() == pytest.approx(0.05, rel=0.05)  # stationary std = variability by construction


# ----------------------------------------------------------------------------- wind
def test_wind_bounds_vector_magnitude_and_wrap():
    m = WindModel(base_speed=3.0, speed_std=4.0, direction_drift_deg=170, seed=3)
    m.reset(seed=3)
    for _ in range(3000):
        s = m.step()
        assert s.speed >= m.min_speed
        assert 0.0 <= s.direction_deg < 360.0
        assert np.hypot(*s.wind_vector) == pytest.approx(s.total_speed, rel=1e-9)


def test_wind_direction_convention():
    north, _ = wind_state(direction=0.0)
    east, _ = wind_state(direction=90.0)
    assert north.wind_vector[0] == pytest.approx(0.0, abs=1e-9) and north.wind_vector[1] < 0  # y points south
    assert east.wind_vector[0] > 0 and east.wind_vector[1] == pytest.approx(0.0, abs=1e-9)


def test_energy_multiplier_signs():
    _, m = wind_state(speed=12.0, direction=90.0)  # blowing toward east
    tail = m.get_energy_multiplier((1, 0))
    head = m.get_energy_multiplier((-1, 0))
    cross = m.get_energy_multiplier((0, 1))
    assert tail < 1.0 < head
    assert cross == pytest.approx(1.0, abs=1e-9)
    assert m.get_energy_multiplier((0, 0)) == 1.0


def test_gust_decay_without_new_gusts():
    m = WindModel(base_speed=5, gust_probability=0.0, gust_decay=0.5, seed=0)
    m.reset(seed=0)
    m.current_gust = 8.0
    m.step()
    assert m.current_gust == pytest.approx(4.0)
    m.step()
    assert m.current_gust == pytest.approx(2.0)


# ----------------------------------------------------------------------------- flood
GRID = np.zeros((9, 9), dtype=np.int8)


def _flood(**kw):
    f = DynamicFloodModel(9, 9, initial_flood_ratio=0.0, **kw)
    f.reset(GRID, seed=0)
    f.flood_grid[:] = 0
    return f


def test_hazard_probability_monotone_in_neighbours_and_rain():
    f = _flood(expansion_rate=0.05, wind_influence=0.0)
    g1 = np.zeros((9, 9), dtype=np.int32)
    g1[4, 3] = 1
    g2 = g1.copy()
    g2[3, 4] = 1
    p1 = f.transition_probabilities(g1, GRID, None, 0.5)[4, 4]
    p2 = f.transition_probabilities(g2, GRID, None, 0.5)[4, 4]
    assert 0.0 < p1 < p2
    assert f.transition_probabilities(g1, GRID, None, 0.9)[4, 4] > p1
    p = f.transition_probabilities(g2, GRID, None, 1.0)
    assert p.min() >= 0.0 and p.max() <= f.max_expansion_probability


def test_wind_pushes_flood_toward_downwind_candidate():
    f = _flood(expansion_rate=0.02, wind_influence=0.3)
    g = np.zeros((9, 9), dtype=np.int32)
    g[4, 4] = 1
    ws, _ = wind_state(speed=15.0, direction=90.0)  # toward east
    p = f.transition_probabilities(g, GRID, ws, 0.5)
    east, west, north = p[4, 5], p[4, 3], p[3, 4]
    assert east > north > 0 and west == pytest.approx(north)


def test_flood_is_expansion_only_and_blocked_never_floods():
    grid = load_map("manhattan32").grid_map
    f = DynamicFloodModel(32, 32, initial_flood_ratio=0.03, expansion_rate=0.06, seed=4)
    f.reset(grid, seed=4)
    prev = f.flood_grid.copy()
    for _ in range(40):
        cur = f.step(grid)
        assert np.all(cur >= prev)
        assert not np.any(cur[grid != 0])
        prev = cur


def test_flood_seed_reproducibility():
    grid = load_map("manhattan32").grid_map
    runs = []
    for seed in (5, 5, 6):
        f = DynamicFloodModel(32, 32, initial_flood_ratio=0.03, expansion_rate=0.06, seed=seed)
        f.reset(grid, seed=seed)
        for _ in range(20):
            f.step(grid)
        runs.append(f.flood_grid.copy())
    assert np.array_equal(runs[0], runs[1]) and not np.array_equal(runs[0], runs[2])


def test_hazard_to_probability_conversion():
    lam = np.linspace(0, 5, 50)
    p = 1 - np.exp(-lam)
    assert p[0] == 0.0 and np.all(np.diff(p) > 0) and p.max() < 1.0


# ----------------------------------------------------------------------------- forecast
def test_forecast_bounds_rules_and_causality():
    grid = load_map("manhattan32").grid_map
    flood = np.zeros_like(grid, dtype=np.int32)
    free = np.argwhere(grid == 0)
    flood[tuple(free[:30].T)] = 1
    ws, _ = wind_state()
    fm = ForecastModel(horizon_steps=8, alpha_fc=0.7)
    a = fm.generate_forecast(flood, grid, ws, 0.6)
    seq = a.flood_prob_sequence
    assert seq.shape == (8, 32, 32) and seq.min() >= 0 and seq.max() <= 1
    assert np.all(seq[:, flood == 1] == 1.0) and np.all(seq[:, grid != 0] == 0.0)
    assert np.all(np.diff(seq, axis=0) >= -1e-6)  # non-decreasing in lead time
    # Deterministic function of current information only.
    b = ForecastModel(horizon_steps=8, alpha_fc=0.7, seed=999).generate_forecast(flood, grid, ws, 0.6)
    np.testing.assert_array_equal(a.flood_prob_sequence, b.flood_prob_sequence)


def test_reliability_blend():
    fm = ForecastModel(horizon_steps=2, alpha_fc=0.25)
    p = fm.reliability_engine.calibrate_probability(np.full((3, 3), 0.8, dtype=np.float32))
    assert p[0, 0] == pytest.approx(0.25 * 0.8 + 0.75 * 0.05)


# ----------------------------------------------------------------------------- payload
def _decay(ptype, wind=None, action=2, prev=(1, 0)):
    m = PayloadStabilityModel(ptype, 0.8 if ptype == "FRAGILE" else 0.4)
    ws = wind or wind_state(0.0, 0.0)[0]
    from rescue_route.env.rescue_route_env import ACTIONS

    return 1.0 - m.step(action, ACTIONS[action], prev, ws).stability


def test_payload_monotonicity():
    calm = _decay("FRAGILE", wind_state(2.0, 0.0)[0])          # moving east, wind from the side
    strong = _decay("FRAGILE", wind_state(10.0, 0.0)[0])
    gusty = _decay("FRAGILE", wind_state(10.0, 0.0, gust=6.0)[0])
    assert calm < strong < gusty
    straight = _decay("FRAGILE", action=2, prev=(1, 0))
    turn90 = _decay("FRAGILE", action=0, prev=(1, 0))
    reverse = _decay("FRAGILE", action=3, prev=(1, 0))
    assert straight < turn90 < reverse
    assert _decay("STANDARD", wind_state(10.0, 0.0)[0]) < strong


def test_payload_hover_recovery_and_bounds():
    m = PayloadStabilityModel("FRAGILE", 0.8)
    m.stability = 0.5
    ws = wind_state()[0]
    s = m.step(8, (0, 0), (1, 0), ws)
    assert s.stability == pytest.approx(0.54)
    m.stability = 0.1  # catastrophic: no recovery
    assert m.step(8, (0, 0), (1, 0), ws).stability == pytest.approx(0.1)
    m.stability = 0.01
    for _ in range(20):
        st = m.step(5, (-1, -1), (1, 1), wind_state(20, 45, 10)[0])
    assert 0.0 <= st.stability <= 1.0 and st.catastrophic_damage and st.is_damaged
