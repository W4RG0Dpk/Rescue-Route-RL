# RescueRoute — Source-audited formula inventory (Phase 0)

Audited from the original repository (`old/Rescue-Route-main`, Environment V1), before any formula was changed.
Every V1 equation below is reproduced bit-for-bit by this repository: `tests/test_v1_equivalence.py` replays full
episodes against the original code (with `scenario_version="v1"` and `legacy_wind_obs=True`) and requires identical
rewards, terminations, flood grids and observations.

Status column: **Retained** (unchanged) · **Fixed** (clear defect corrected, V1 behaviour still available) ·
**Rebalanced** (formula unchanged, scenario range changed) · **Calibrated alternative** (new ERA5-fitted V2 option; V1 untouched).

## Audit answers to the report's explicit questions (§16, prompt item 5)

| Question | Answer from source |
|---|---|
| Does flood recession / drainage exist? | **No.** Flooded cells stay flooded for the whole episode (`hazards/flood.py`, "Flooded cells remain flooded"). The model is expansion-only. |
| Wind direction convention? | **Vector "toward"**: 0° = north, 90° = east; `vx = S·sin θ`, `vy = −S·cos θ` in grid coordinates (x east, y south). ERA5 "from" directions are converted once, as θ_sim = θ_met + 180° (`validation/data.py`). |
| How is rain intensity advanced? | Clipped Gaussian AR(1) on a **normalised, unitless** intensity in [0, 1], updated inside the flood model's `step()` **before** flood expansion, using the flood model's RNG. |
| Observation contract? | `spatial_tensor (7, 17, 17)` in [0, 1] + `vector_state (15,)` in [−1, 1]; 9 discrete actions (0 N, 1 S, 2 E, 3 W, 4 NE, 5 NW, 6 SE, 7 SW, 8 Hover). Confirmed. |

## 1. Weather

| # | Quantity | Exact source expression | Source | Units / range | Parameters (source) | RNG / distribution | Timing | Tests | Status |
|---|---|---|---|---|---|---|---|---|---|
| W1 | Wind speed | `V ← max(V_min, μ + ρ(V − μ) + ε)`, `ε ~ N(0, σ√(1−ρ²))` | `weather/wind.py::WindModel.step` | m/s, ≥ 0.5 | μ = `wind_base_speed`, σ = `wind_speed_std`, ρ = `wind_correlation` (mission) | wind RNG, Normal | every env step, after the move | `test_wind_bounds_vector_magnitude_and_wrap` | Retained; ERA5 validation **supports** it (speed level, spread, lag-1 ACF) |
| W2 | Wind direction | `θ ← (θ + U(−d, d)) mod 360` | same | degrees, "toward" | d = `wind_direction_drift_deg` = 15 | wind RNG, Uniform | same | `test_wind_direction_convention` | Retained in V1; ERA5 shows no prevailing wind (R = 0.10 vs 0.82) → V2 uses VAR(1) on (u, v) |
| W3 | Gust | with prob. p: `G ← U(g_min, g_max)`, else `G ← 0.5·G` (`G < 1e−6 → 0`) | same | m/s, additive to V | p, g_min, g_max (mission); decay 0.5 (constructor) | wind RNG, Bernoulli + Uniform | same | `test_gust_decay_without_new_gusts` | Retained in V1; V2: continuous speed-dependent AR(1) gust excess |
| W4 | Wind vector | `S = V + G`; `(vx, vy) = (S sin θ, −S cos θ)` | `WindModel.get_state` | m/s | — | — | — | `test_wind_direction_convention` | Retained |
| W5 | Energy multiplier | `clip(1 − 0.20·(m̂·ŵ)·clip(S/15, 0, 1), 0.75, 1.40)`; hover → 1 | `WindModel.get_energy_multiplier` | dimensionless | constants 0.20, 15, 0.75, 1.40 (code) | — | at the action | `test_energy_multiplier_signs` (tail < 1 < head, cross = 1) | Retained |
| R1 | Rain intensity | `R ← clip(μ + ρ(R − μ) + ε, 0, 1)`, `ε ~ N(0, σ√(1−ρ²))`; reset `R₀ = clip(μ + N(0, σ), 0, 1)` | was `hazards/flood.py::_update_rain_intensity`; now `weather/rain.py::RainModelV1` (same RNG stream) | normalised [0, 1], **no physical unit** | μ = `rain_intensity` ~ U(0.15, 0.90), σ = 0.05 + 0.20 μ, ρ = 0.85 (mission) | flood RNG, Normal | start of flood step | `test_rain_bounds_and_no_nan`, `test_rain_persistence_matches_rho_without_clipping` | Retained in V1; ERA5: persistence supported, spells/intensity shape not → V2 two-scale latent-Gaussian |

## 2. Flood (`hazards/flood.py::DynamicFloodModel`)

| # | Quantity | Exact source expression | Units / range | Parameters | RNG | Timing | Tests | Status |
|---|---|---|---|---|---|---|---|---|
| F1 | Initial seeds | `n = max(1, round(W·H·r₀))` cells, sampled without replacement from free cells excluding start/target | binary grid | r₀ = `flood_initial_ratio` | flood RNG `choice` | reset | `test_flood_seed_reproducibility` | Formula retained; **rebalanced** range ×0.5 (v1.1) |
| F2 | Spread probability (≥ 1 flooded 4-neighbour) | `p = clip(k·n·(0.5 + 0.75 R) + k_w·max(0, ā)·clip(S/15, 0, 1), 0, 0.60)`; `ā` = mean over flooded neighbours of `d̂(j→c)·ŵ` | probability | k = `flood_expansion_rate`, k_w = `flood_wind_influence` = 0.05; 0.5, 0.75, 0.60 code constants | — | after rain update | `test_hazard_probability_monotone_in_neighbours_and_rain`, `test_wind_pushes_flood_toward_downwind_candidate` | Formula retained; k **rebalanced** ×0.35 (v1.1) |
| F3 | Spontaneous flooding (no flooded neighbour) | `p = p_s·R` | probability | p_s = 0.001 | — | same | — | Retained |
| F4 | Transition draw | `F[t+1](c) = 1 if U_c < p_c`, one `U(0,1)` per candidate in row-major order | binary | — | flood RNG `random` | same | `test_v1_equivalence` | Retained (vectorised, same draw order) |
| F5 | Persistence | flooded cells never dry; blocked cells never flood | — | — | — | — | `test_flood_is_expansion_only_and_blocked_never_floods` | Retained (documented as expansion-only simplification) |
| F6 | Wind sign | water moves from flooded neighbour j to candidate c; wind pushing toward c raises p | — | — | — | — | `test_wind_pushes_flood_toward_downwind_candidate` | Retained, labelled a **directional heuristic**, not validated |

**Why rebalanced (v1.1).** Measured with a clairvoyant oracle (true future flood, exact time-expanded reachability), only
**62 %** of `normal` and **0 %** of `dynamic_flood` V1 missions were solvable at all, and ~43 % of free airspace was flooded
by step 40. Scaling the two scenario ranges (k ×0.35, r₀ ×0.5) gives 94 % / 60 % oracle feasibility. Only the mission
generator's ranges changed; F2–F5 are untouched. `scenario_version="v1"` reproduces V1.

## 3. Forecast (`forecast/forecast_model.py`, `forecast/reliability.py`)

| # | Quantity | Exact source expression | Notes | Tests | Status |
|---|---|---|---|---|---|
| P1 | Wind forecast | `V̂ₖ = max(0, μ + ρᵏ(V_t − μ))`, `θ̂ₖ = θ_t` | sustained speed only (no gust) | — | Retained |
| P2 | Flood probability propagation | `P_nb = clip(Σⱼ p(j)/4, 0, 1)`; `a = clip(Σⱼ p(j) d̂·ŵ / (4P_nb), −1, 1)`; `λ = 4k·P_nb·(0.5 + 0.75R) + k_w·max(0, a)·s + p_s·R`; `q = clip(1 − e^(−λ), 0, 0.95)`; `pₖ = pₖ₋₁ + (1 − pₖ₋₁)q`; blocked → 0; p ≥ 0.999999 → 1 | **hazard-rate conversion with Δt = 1 step** (no extra time factor); mean-field over neighbours | `test_forecast_bounds_rules_and_causality` | Retained (vectorised) |
| P3 | Reliability blend | `p_vis = α·p_model + (1 − α)·0.05`; currently flooded → 1; blocked → 0 | α = `forecast_reliability` ~ U(0.5, 0.95) | `test_reliability_blend` | Retained |
| P4 | Causality | forecast reads current flood, static map, current wind and current rain only | no RNG is ever used | `test_forecast_bounds_rules_and_causality` | Verified |

Calibration check (normal profile, lead 10 steps): mean forecast p = 0.044 vs observed 0.023 (over-forecast, mostly from the
0.05 prior), Brier 0.0207 vs climatology 0.0228, so the skill is positive. For dynamic_flood at lead 40: p = 0.568 vs 0.528.

## 4. Payload (`payload/stability.py::PayloadStabilityModel.step`)

| # | Quantity | Exact source expression | Parameters | Tests | Status |
|---|---|---|---|---|---|
| L1 | Turn severity | `τ = clip((1 − m̂·m̂_prev)/2, 0, 1)` (0 straight, 0.5 at 90°, 1 reversal; 0 if hover or no heading) | — | `test_payload_monotonicity` | Retained |
| L2 | Crosswind | `|−m̂_y·v_x + m̂_x·v_y|` (total wind incl. gust) | — | same | Retained |
| L3 | Moving update | `s ← clip(s − [α_w·clip(c/10, 0, 2)·σ_w + β_g·clip(G/8, 0, 2)·σ_g + γ·τ·σ_t], 0, 1)` | α_w = 0.025, β_g = 0.05, γ = 0.035; σ = (0.35, 0.30, 0.35) STANDARD, (1, 1, 1) FRAGILE | same | Retained |
| L4 | Hover | `s ← min(1, s + 0.04·σ_h)` unless `s ≤ 0.15` (catastrophic) | σ_h = 0.75 / 1.0 | `test_payload_hover_recovery_and_bounds` | Retained |
| L5 | Flags | damaged: `s < s_req` (0.40 STANDARD, 0.80 FRAGILE); catastrophic: `s ≤ 0.15` | — | same | Retained |

## 5. Environment (`env/rescue_route_env.py`)

| # | Item | Source | Status |
|---|---|---|---|
| E1 | Step order | energy (current wind) → payload → move/collision checks → delivery → battery → deadline → wind step → rain/flood step → engulfment → new forecast | Retained |
| E2 | Base energy | cardinal 1, diagonal √2, hover 0.5, × W5 multiplier | Retained |
| E3 | Collisions | boundary, blocked/NFZ, diagonal corner blocked, destination flooded, diagonal corner flooded, engulfment after flood step | Retained |
| E4 | Reward (V1) | −0.05/step; +1.5·Δoctile; −0.5·(1 − s); −0.02·energy; +100 delivery; −50 collision; −30 flood; −50 battery; −50 deadline; −40 payload | Retained as **default** (`reward_version="v1"`) |
| E5 | Reward (v1.1, training only) | flood/collision −60, geodesic progress instead of octile | **New option.** V1's −30 flood vs −50 deadline makes crashing early optimal for an agent expecting to fail (65 % of first-run episodes ended this way) |
| E6 | `vector_state[8:10]` | V1 stored `clip(vx, −1, 1), clip(vy, −1, 1)` of the m/s wind vector (≈ sign) although documented as direction components | **Fixed** → unit vector; `legacy_wind_obs=True` restores V1 |
| E7 | Mission targets | farthest 20 % of the start's 8-connected component; octile distance; battery = 8·D·U(1.35, 2.2); deadline = ⌈D_steps·U(1.35, 2.0)⌉ | Retained (training curriculum may draw nearer targets; evaluation always uses V1's rule) |

## 6. New components (not in V1)

| Component | Equation | Source |
|---|---|---|
| Recoverability margin | `M_rec = min((τ_D − T̂_D)/τ₀, (B_t − Ê_D − Ê_R − B_reserve)/B₀)`, B_reserve = 0.1·B₀, τ₀ = deadline, B₀ = initial battery | `safety/evaluator.py` |
| Forecast recalibration (evaluator only) | `p_cal = g_{lead, α}(p_visible)`, isotonic (PAV) per lead-time × α bucket, fitted on training seeds 300,000–300,419; pooled Brier 0.115 → 0.103 | `forecast/calibration.py`, `configs/forecast_calibration.json` |
| Unrecoverability | `P̂_unrec = (1/K) Σₖ 1[M_rec^(k) < 0]`, K = 16 (48 to confirm an empty safe set) | same |
| Forecast-conditioned flood futures | `Fₖ[h](c) = 1[Uₖ(c) < p_h(c)]`, Uₖ = Φ(smoothed Gaussian field): monotone and exactly the forecast marginals | same |
| Safe set | `A_safe = {a : immediate ok ∧ P̂_unrec ≤ δ}`; graded fallback (ε = 0.1 indifference margin, abort if min P̂ ≥ δ_abort; δ and δ_abort tuned on validation seeds) | `safety/shield.py` |
| V2 rain | `Z = √w Z_f + √(1−w) Z_s` (two AR(1)s); `P = 0` if `Z ≤ c`, else `P_wet + F_Γ⁻¹((Φ(Z) − Φ(c))/(1 − Φ(c)))`; flood driver `clip(API/P_ref, 0, 1)` | `weather/calibrated.py` |
| V2 wind | `w_t = m(h_t) + d_t`, `d_{t+1} = A d_t + e_t`; gust excess `max(0, a + b|w| + η_t)` | same |
