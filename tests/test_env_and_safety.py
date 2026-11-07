"""Environment contract, planning, safety-layer and no-leakage tests."""

import numpy as np
import pytest

from rescue_route.env import SCENARIO_PROFILES, RescueRouteEnv, generate_mission
from rescue_route.map import load_map
from rescue_route.planning.astar import AStarDynamic, astar, octile
from rescue_route.planning.geodesic import GeodesicField
from rescue_route.planning.reachability import earliest_arrival, true_flood_future
from rescue_route.safety.evaluator import EvaluatorConfig, ForecastConsequenceEvaluator
from rescue_route.safety.immediate import immediate_check
from rescue_route.safety.shield import LAND, SafetyShield


def test_observation_contract_and_spaces():
    env = RescueRouteEnv()
    obs, info = env.reset(seed=11)
    assert obs["spatial_tensor"].shape == (7, 17, 17) and obs["vector_state"].shape == (15,)
    assert env.observation_space.contains(obs) and env.action_space.n == 9
    for a in [2, 2, 6, 8, 0]:
        obs, r, term, trunc, info = env.step(a)
        assert env.observation_space.contains(obs) and np.isfinite(r)
        if term:
            break
    # wind direction components are a unit vector after the V1.1 fix
    assert np.hypot(obs["vector_state"][8], obs["vector_state"][9]) == pytest.approx(1.0, abs=1e-5)


def test_same_seed_same_trajectory_different_seed_differs():
    def run(seed):
        env = RescueRouteEnv()
        env.reset(seed=seed)
        out = []
        for a in [8] * 25:
            o, r, d, _, i = env.step(a)
            out.append((r, i["flooded_cells_count"], i["wind_speed"]))
            if d:
                break
        return out

    assert run(5) == run(5) and run(5) != run(6)


def test_missions_valid_for_every_profile():
    m = load_map("manhattan32")
    for profile in SCENARIO_PROFILES:
        for seed in range(5):
            ms = generate_mission(m, seed=seed, profile=profile)
            assert m.is_base(*ms.start_pos) and not m.is_blocked(*ms.target_pos)
            assert ms.deadline_steps > 0 and ms.initial_battery > 0


def test_battery_monotone_and_time_progresses():
    env = RescueRouteEnv()
    env.reset(seed=3)
    b, t = env.current_battery, env.current_step
    for _ in range(10):
        _, _, d, _, info = env.step(8)
        assert info["battery_remaining"] < b and info["step"] == t + 1
        b, t = info["battery_remaining"], info["step"]
        if d:
            break


def test_static_action_mask_matches_geometry():
    env = RescueRouteEnv()
    env.reset(seed=1)
    for _ in range(30):
        mask = env.static_action_mask()
        x, y = env.drone_pos
        for a in range(9):
            from rescue_route.env.rescue_route_env import ACTIONS

            dx, dy = ACTIONS[a]
            if not mask[a]:
                assert env.map_data.is_blocked(x + dx, y + dy) or (dx and dy and (env.map_data.is_blocked(x + dx, y) or env.map_data.is_blocked(x, y + dy)))
        _, _, d, _, _ = env.step(int(np.flatnonzero(mask)[0]))
        if d:
            break


def test_astar_path_valid_and_geodesic_agrees():
    m = load_map("manhattan32")
    ms = generate_mission(m, seed=7)
    path = astar(m.blocked_mask, ms.start_pos, ms.target_pos)
    assert path[0] == ms.start_pos and path[-1] == ms.target_pos
    cost = sum(octile(a, b) for a, b in zip(path, path[1:]))
    geo = GeodesicField(m.blocked_mask).distances(ms.target_pos, np.zeros_like(m.grid_map))
    assert geo[ms.start_pos[1], ms.start_pos[0]] == pytest.approx(cost)


def test_reachability_matches_executed_astar_with_true_future():
    """If A* delivers in n steps, the clairvoyant earliest arrival can be no later than n."""
    env = RescueRouteEnv()
    checked = 0
    for seed in range(2_000_000, 2_000_010):
        env.reset(seed=seed)
        D = env.mission.deadline_steps
        floods = true_flood_future(env, D + 1)
        goal = np.zeros_like(floods[0])
        goal[env.mission.target_pos[1], env.mission.target_pos[0]] = True
        arr, _ = earliest_arrival(env.drone_pos, goal, floods, ~env.map_data.blocked_mask, D)
        planner = AStarDynamic()
        while True:
            _, _, done, _, info = env.step(planner.act(env))
            if done:
                break
        if info["is_success"]:
            assert 0 < arr <= info["step"]
            checked += 1
    assert checked >= 3


def test_true_flood_future_does_not_disturb_env():
    env = RescueRouteEnv()
    env.reset(seed=9)
    before = env.flood_model.rng.bit_generator.state
    fut = true_flood_future(env, 10)
    assert env.flood_model.rng.bit_generator.state == before
    for t in range(1, 6):
        env.step(8) if not env.done else None
        if env.done:
            break
        assert np.array_equal(env.flood_model.flood_grid.astype(bool), fut[t])


def test_evaluator_has_no_access_to_hidden_future():
    """Same PlannerView + same evaluator seed -> identical assessment, whatever the env's hidden RNG."""
    env_a, env_b = RescueRouteEnv(profile="dynamic_flood"), RescueRouteEnv(profile="dynamic_flood")
    env_a.reset(seed=21)
    env_b.reset(seed=21)
    env_b.flood_model.rng = np.random.default_rng(12345)  # different hidden future
    env_b.wind_model.rng = np.random.default_rng(54321)
    va, vb = env_a.get_planner_view(), env_b.get_planner_view()
    ra = ForecastConsequenceEvaluator(EvaluatorConfig(num_samples=8), seed=1).evaluate(va)
    rb = ForecastConsequenceEvaluator(EvaluatorConfig(num_samples=8), seed=1).evaluate(vb)
    for a in range(9):
        assert ra[a].p_unrecoverable == rb[a].p_unrecoverable
        assert ra[a].immediate_ok == rb[a].immediate_ok


def test_evaluator_safe_in_benign_world_and_flags_immediate_hazards():
    env = RescueRouteEnv()
    env.reset(seed=2_000_001)
    v = env.get_planner_view()
    import dataclasses

    calm = dataclasses.replace(v, forecast_sequence=np.repeat(v.current_flood[None].astype(np.float32), 10, 0),
                               scenario_params={**v.scenario_params, "flood_expansion_rate": 0.0, "flood_spontaneous_probability": 0.0})
    res = ForecastConsequenceEvaluator(EvaluatorConfig(num_samples=4), seed=0).evaluate(calm)
    ok = [a for a in res if res[a].immediate_ok]
    assert ok and all(res[a].p_unrecoverable == 0.0 for a in ok)
    for a in range(9):
        if not res[a].immediate_ok:
            assert not immediate_check(calm, a)[0]


def test_shield_modes_return_legal_decisions():
    env = RescueRouteEnv(profile="compound")
    for mode in ("none", "immediate", "deterministic", "probabilistic"):
        env.reset(seed=2_000_002)
        sh = SafetyShield(mode, num_samples=6, confirm_samples=8)
        for _ in range(6):
            probs = np.full(9, 1 / 9)
            d = sh.decide(probs, env.get_planner_view())
            assert d.action == LAND or 0 <= d.action <= 8
            if d.action == LAND:
                env.declare_landing()
                break
            if mode != "none":
                assert immediate_check(env.get_planner_view(), d.action)[0] or d.source == "unrecoverable"
            _, _, done, _, _ = env.step(d.action)
            if done:
                break


def test_v2_weather_runs_and_is_bounded():
    pytest.importorskip("scipy")
    from rescue_route.weather.calibrated import load_calibrated_params

    try:
        load_calibrated_params()
    except FileNotFoundError:
        pytest.skip("run scripts/validate_weather.py first")
    env = RescueRouteEnv(weather_model="v2")
    env.reset(seed=4)
    for _ in range(30):
        _, _, d, _, info = env.step(8)
        assert info["wind_speed"] >= 0 and 0 <= info["rain_intensity"] <= 1
        if d:
            break
