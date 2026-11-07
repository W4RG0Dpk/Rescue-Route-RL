# RescueRoute — Complete Project Guide

This guide explains every folder and file in the project, how the modules fit together, how the end-to-end
workflow runs, and why each algorithm was chosen over its alternatives (with the measured evidence).

---

## 1. What the project does, in one paragraph

A single drone (UAV) must fly an emergency medical payload from a base to a hospital across a flooding city
(the 32×32 *Manhattan32* grid). Wind, rain and floods change randomly while it flies. It must arrive before a
deadline, keep a fragile payload stable, not run out of battery, and **still be able to land safely afterwards**.
The project builds (1) a simulator of this world, (2) checks the simulator's weather against real Mumbai monsoon
data, (3) trains neural-network pilots with reinforcement learning, and (4) adds a **recoverability safety layer**
that imagines many possible futures before every move and refuses moves that are likely to doom the mission.
Everything is evaluated on 9,900 held-out missions, written up as an IEEE Access paper, and shown in an
interactive 3-D-style web demo.

**Research question.** *Do forecast-based, look-ahead recoverability checks reduce future mission failures
without hurting on-time delivery, compared with the usual one-step safety check?* Result: UAV losses fall from
31.1 % to 6.1 %, and delivered-and-recoverable missions do not decrease (+2.1 pp).

---

## 2. The workflow, end to end

```
 ┌────────────────────┐   ┌──────────────────────┐   ┌───────────────────────┐   ┌────────────────────┐
 │ 1. Simulator       │──▶│ 2. Weather validation│──▶│ 3. Forecast           │──▶│ 4. Pre-training     │
 │ (env, weather,     │   │ ERA5 Mumbai data,     │   │ recalibration         │   │ teacher → student   │
 │  flood, payload,   │   │ fit + test models     │   │ (isotonic, training   │   │ (behaviour cloning  │
 │  forecast)         │   │ → V2 weather config   │   │  seeds)               │   │  + DAgger)          │
 └────────────────────┘   └──────────────────────┘   └───────────────────────┘   └─────────┬──────────┘
                                                                                             ▼
 ┌────────────────────┐   ┌──────────────────────┐   ┌───────────────────────┐   ┌────────────────────┐
 │ 8. Paper + demo    │◀──│ 7. Reports           │◀──│ 6. Evaluation          │◀──│ 5. PPO fine-tuning  │
 │ IEEE Access LaTeX, │   │ evaluation, per-     │   │ tune δ on validation   │   │ MLP, CNN, CNN+LSTM, │
 │ web Mission Control│   │ profile, figures     │   │ seeds → test 9,900 eps │   │ CNN+LSTM+risk       │
 └────────────────────┘   └──────────────────────┘   └───────────────────────┘   └────────────────────┘
```

| Step | Command | Reads | Writes | Time* |
|---|---|---|---|---|
| Tests | `python -m pytest` | code, `old/` (for equivalence) | – | 15 s |
| 2. Weather validation | `python scripts/validate_weather.py` | Open-Meteo API (cached in `data/weather`) | `reports/weather_validation/*`, `configs/weather_v2_calibrated.json` | < 1 min |
| 3. Forecast calibration | `python scripts/calibrate_forecast.py` | simulator (training seeds) | `configs/forecast_calibration.json` | 3 min |
| 4. Pre-training | `python scripts/pretrain.py` | simulator + A* teacher | `outputs/models/bc_*.pt`, `bc_log.json` | 20 min |
| 5. PPO training | `python scripts/train.py --variant all` | `bc_*.pt` | `outputs/models/ppo_*/` | ~1.5 h |
| 6. Evaluation | `python scripts/evaluate.py --stage all` | trained models, calibration | `reports/evaluation/*` (+ `profile_report.md`) | ~60 min |
| 7. Training plot | `python scripts/plot_training.py` | `progress.csv` files | `reports/training/training_curves.png` | 5 s |
| 8a. Demo bundle | `python scripts/export_demo.py` | models, reports | `src/rescue_route/demo/web/data/*`, `docs/showcase_mission.png` | 2 min |
| 8b. Live demo | `python scripts/run_demo.py` | everything above | web server on http://127.0.0.1:8765 | – |
| 8c. Paper figures | `python scripts/make_paper_figures.py` | reports, models, data | `IEEE Report/figures/*`, `paper_numbers.json` | 1 min |

*Measured on a 32-thread CPU with an RTX 4060 laptop GPU.

**Seed discipline (prevents cheating).** Training uses seeds in `[0, 1,000,000)`; threshold tuning uses
validation seeds `1,000,000+i`; final results use test seeds `2,000,000+i`. Forecast calibration uses training
seeds `300,000–300,419`. Test seeds are never used for any choice.

---

## 3. Folder map

```
RL Project/
├── old/Rescue-Route-main/          Your original V1 code (kept untouched; used as a reference in tests)
├── .claude/launch.json             Lets the desktop app start the demo server in its browser pane
└── Reinforcement-Learning-Project/ ← the project
    ├── README.md                   Overview, results, how to run
    ├── requirements.txt            Python dependencies
    ├── pyproject.toml              Package metadata (pip install -e .)
    ├── .gitignore                  Files git should ignore (logs, caches)
    ├── IEEE Report/                The paper (LaTeX + figures)
    ├── IEEE Report.zip             A zip of the paper folder (created outside these steps)
    ├── configs/                    Generated parameter files (V2 weather, forecast calibration)
    ├── data/                       Map images and cached ERA5 weather data
    ├── docs/                       Formula inventory, this guide, showcase figure
    ├── outputs/                    Trained models, training logs, temporary build files
    ├── reports/                    Generated reports: evaluation, weather validation, training
    ├── scripts/                    Command-line entry points (one per workflow step)
    ├── src/rescue_route/           The Python package (all the real logic)
    └── tests/                      Automated tests (32)
```

---

## 4. File-by-file reference

### 4.1 Top level

| File | Purpose |
|---|---|
| `README.md` | Project summary, results table, quick start, design decisions, limitations. |
| `requirements.txt` | numpy, scipy, pandas, matplotlib, pillow, gymnasium, torch, statsmodels, requests, flask, pytest. |
| `pyproject.toml` | Makes `src/rescue_route` an installable package; tells pytest where tests live. |
| `.gitignore` | Excludes caches, logs and very large per-episode training CSVs from git. |

### 4.2 `data/`

| File | Purpose |
|---|---|
| `maps/manhattan32/map.png` | The 32×32 city. One pixel = one cell. Colours: red = no-fly zone, yellow = tower inside a no-fly zone, green = low building the drone can fly over (rooftop landing), blue = base/helipad, black = free air. |
| `maps/manhattan32/config.json` | Original benchmark settings that came with the map (from the UAV data-harvesting benchmark). Kept for reference. |
| `maps/urban50/*` | A larger 50×50 map, supported by the loader but not used in experiments. |
| `weather/era5_mumbai_2021-01-01_2023-12-31.csv` | Cached hourly ERA5 data for Mumbai: precipitation, rain, wind speed, direction and gusts (26,280 hours). |
| `weather/...meta.json` | Where the data came from, licence, grid cell, units, download time. |

### 4.3 `configs/` (generated — do not edit by hand)

| File | Produced by | Contents |
|---|---|---|
| `weather_v2_calibrated.json` | `validate_weather.py` | Fitted parameters of the better weather models (two-scale latent rain, diurnal VAR wind). Used when the env runs with `weather_model="v2"`. |
| `forecast_calibration.json` | `calibrate_forecast.py` | 18 isotonic calibration curves (6 lead-time × 3 reliability buckets) and their Brier-score statistics. Used by the safety evaluator. |

### 4.4 `docs/`

| File | Purpose |
|---|---|
| `FORMULA_INVENTORY.md` | Every equation in the simulator, its source file, parameters, random-number use, tests, and whether it was kept, fixed, rebalanced or replaced. This was the "Phase 0 audit" your handoff report asked for. |
| `PROJECT_GUIDE.md` | This document. |
| `showcase_mission.png` | Side-by-side map of one mission flown with the immediate filter vs the full shield. |

### 4.5 `scripts/` (what you run)

| Script | What it does |
|---|---|
| `validate_weather.py` | Downloads/loads ERA5, fits all weather models on 2021–2022, tests on 2023, writes the validation report, figures and the V2 config. Options: `--location`, `--season`, `--p-ref`. |
| `calibrate_forecast.py` | Runs 420 training missions, compares the forecast to what actually flooded, fits isotonic recalibration curves. |
| `pretrain.py` | Teacher (A*) flies 2,500 missions → behaviour cloning → 2 DAgger rounds; saves `bc_mlp.pt`, `bc_cnn.pt`, `bc_cnn_lstm.pt`. |
| `train.py` | PPO fine-tuning of one variant or `all` (MLP, CNN, CNN+LSTM, CNN+LSTM+risk) from the cloned start. |
| `evaluate.py` | `--stage tune`: grid-search the shield thresholds on validation seeds. `--stage test`: every method × every profile × 100 test seeds, then writes all reports. |
| `plot_training.py` | Training-curve figure from each model's `progress.csv`. |
| `export_demo.py` | Picks showcase missions from the test results, records them for 4 methods, bundles benchmark + weather results for the web demo, draws the showcase figure. |
| `run_demo.py` | Flask web server for the demo with a live simulation API (`/api/run`, `/api/suggest`, `/api/meta`). |
| `make_paper_figures.py` | Builds all paper figures (white background, vector PDF, Arial) and `paper_numbers.json`. |

### 4.6 `src/rescue_route/` — the package

#### `map/`
| File | Purpose |
|---|---|
| `map_loader.py` | Reads `map.png`, turns colours into masks (blocked, no-fly, fly-over, base, free) and a `MapData` object with helpers like `is_blocked(x, y)`. Unchanged from V1. |

#### `env/` — the simulator core
| File | Purpose |
|---|---|
| `mission.py` | Generates one mission from a seed: start base, far hospital target, battery, deadline, payload type, weather/flood parameters. Adds the 11 **stress profiles** (normal, low battery, tight deadline, strong wind, dynamic flood, fragile, wind+flood, low battery+deadline, compound, all combined, low forecast reliability) and the **scenario versions** (`v1` original ranges, `v1.1` rebalanced flood ranges). Also the octile-distance helper and a cached connected-components search. |
| `rescue_route_env.py` | The Gymnasium environment `RescueRouteEnv`. `reset()` builds the mission and all models; `step(action)` applies energy → payload → move/collision → delivery → battery → deadline → weather/flood update → engulfment → new forecast, returns observation, reward, done, info. Also: observation building (7×17×17 tensor + 15-value vector), static action mask, `get_planner_view()` (safe information for the shield), `clone()`, `declare_landing()` (fallback only), training-reward option `v1.1`, risk penalty option, V2 weather option. |

#### `weather/`
| File | Purpose |
|---|---|
| `wind.py` | V1 wind: AR(1) speed, random-walk direction, Bernoulli gusts with decay; helpers for along-wind, crosswind and the energy multiplier. Unchanged from V1. |
| `rain.py` | V1 rain intensity (clipped Gaussian AR(1)), extracted from the old flood file so it can be validated on its own. Same random-number stream → identical results. |
| `calibrated.py` | V2 (ERA5-fitted) drop-in replacements: `CalibratedWindModel` (VAR(1) on wind vector + daily cycle + gust excess) and `CalibratedRainModel` (two-scale latent-Gaussian rain + antecedent-rain flood driver). |

#### `hazards/`
| File | Purpose |
|---|---|
| `flood.py` | The flood cellular automaton, vectorised with NumPy. Each dry cell next to water floods with a probability depending on flooded neighbours, rain and wind; isolated cells can flood spontaneously; water never recedes. Bit-identical to V1 (same random draws, same order) but ~1.5× faster. Also records the step each cell flooded (used by the demo). |

#### `payload/`
| File | Purpose |
|---|---|
| `stability.py` | Payload stability (0–1). Drops with crosswind, gusts and sharp turns (fragile payloads 3× more sensitive); recovers +0.04 per hover step unless it has fallen below the catastrophic level 0.15. Unchanged from V1. |

#### `forecast/`
| File | Purpose |
|---|---|
| `forecast_model.py` | The flood forecast the drone sees: propagates flood probabilities 10 steps ahead (mean-field version of the flood rule, hazard → probability via 1−e^−λ) plus a wind forecast. Vectorised, identical to V1. |
| `reliability.py` | Blends the forecast with a background prior: `p = α·p_model + (1−α)·0.05`; currently flooded = 1, blocked = 0. |
| `calibration.py` | NEW. Fits and applies the isotonic recalibration map used by the safety evaluator (fixes the forecast's over-prediction at short lead times). |

#### `planning/`
| File | Purpose |
|---|---|
| `astar.py` | A* shortest path (octile heuristic, no corner cutting) and three planners (static, dynamic, forecast-aware). **Used only as the training teacher** for imitation learning; not a benchmark method. |
| `geodesic.py` | Obstacle- and flood-aware distance to the hospital via SciPy's C Dijkstra; used for the v1.1 training reward. |
| `reachability.py` | Vectorised **time-expanded** search over a sequence of flood maps: earliest arrival and minimum energy, with the exact movement rules. Also `true_flood_future()` (post-hoc only) for oracle feasibility. |

#### `rl/` — learning
| File | Purpose |
|---|---|
| `networks.py` | Neural networks: MLP encoder, CNN encoder, CNN+LSTM recurrent actor-critic; `PolicyRunner` for single-episode inference (keeps LSTM memory, applies action mask); checkpoint loading. |
| `vec_env.py` | Runs many environments in parallel worker processes (e.g. 128 envs in 16 processes) with auto-reset; fails loudly if a worker dies. |
| `ppo.py` | The PPO trainer (feed-forward and recurrent), GAE, static action masking, initialisation from a cloned policy, critic warm-up, decaying KL anchor to the clone, atomic checkpoint saving, CSV logging. |
| `imitation.py` | Behaviour cloning + DAgger: parallel teacher/student roll-outs, episode dataset (keeps sequences intact for the LSTM), cross-entropy training. |

#### `safety/` — the research contribution
| File | Purpose |
|---|---|
| `information.py` | `PlannerView`: a frozen snapshot of **only** the information the shield may use (map, current flood, forecast, observed wind, UAV state). No random-generator state, no future. |
| `immediate.py` | The conventional one-step safety check (no collision, no flooded cell/corner, battery > 0, payload not destroyed, deadline still reachable). |
| `evaluator.py` | The **forecast consequence evaluator**: samples K futures consistent with the (recalibrated) forecast, runs the exact time-expanded min-energy search for every candidate action in every future, adds payload re-stabilisation time and the return-to-landing leg, computes the recoverability margin and P_unrec. |
| `shield.py` | The **recoverability shield**: modes `none / immediate / deterministic / probabilistic`; safe set by δ; confirmation re-sampling; indifference margin ε; abort threshold δ_abort; fallback controller (return to base, rooftop landing, hover). |

#### `evaluation/`
| File | Purpose |
|---|---|
| `controllers.py` | Builds every evaluated method (random, BC student, 4 PPO variants, 3 shielded variants) behind one `decide()` interface; loads models. |
| `runner.py` | Runs one episode; computes post-hoc truth metrics (oracle feasibility, post-delivery recovery on the true future, future failure); records compact traces for the demo. |
| `experiments.py` | Parallel grid runner (method × profile × seed), seed-range constants, Wilson confidence intervals, paired bootstrap, summary tables. |
| `report.py` | `evaluation_report.md` + figures: headline, overall table, per-profile tables, paired comparisons, tuning table. |
| `profile_report.py` | Writes `profile_report.md`: every method × every profile, 14 metrics each, plus discussion. |

#### `validation/` — weather checks against real data
| File | Purpose |
|---|---|
| `data.py` | Downloads ERA5 from Open-Meteo with retries + caching; converts meteorological "wind from" to the simulator's "wind toward"; derives u/v components and gust excess; season filtering; splits into continuous hour runs. |
| `stats.py` | Statistics compared between real and simulated weather: means, spreads, wet fraction, wet/dry spell lengths, quantiles, autocorrelation, direction steadiness, gust persistence. |
| `models.py` | Fits and simulates every candidate model (V1 rain, Markov-Gamma, latent-Gaussian, two-scale latent; V1 wind, VAR, VAR+diurnal). The V1 versions call the simulator's real classes. |
| `pipeline.py` | Orchestrates: load data → fit on training years → simulate 40 runs → compare with test year → residual tests → figures → report → V2 config. |
| `plots.py` | Validation figures (time series, exceedance curves, spells, autocorrelation, wind roses). |
| `report.py` | Writes `weather_validation_report.md`; all numbers and verdicts computed, none typed by hand. |

#### `demo/web/` — Mission Control
| File | Purpose |
|---|---|
| `index.html` | Page layout: tabs (Mission, Compare, Benchmark, Weather validation, How it works), panels, controls. |
| `style.css` | Dark "mission control" visual design, responsive layout. |
| `app.js` | Isometric 2.5-D renderer (buildings, water, rain/wind particles, drone with rotors, hospital beacon, shield risk fan), HUD gauges, timeline, toasts, compare mode, benchmark/weather pages, live-simulation client. |
| `data/demo_data.js` | Recorded showcase missions + benchmark + weather results (generated by `export_demo.py`). |
| `data/figures/*.png` | Weather figures shown on the Weather tab. |

### 4.7 `tests/`
| File | What it guarantees |
|---|---|
| `conftest.py` | Puts `src/` on the path; locates your original V1 code. |
| `test_v1_equivalence.py` | Full episodes replayed against your **original** code: identical rewards, terminations, flood grids, observations. |
| `test_models.py` | Mathematical properties: rain bounds/persistence, wind bounds/direction convention/energy signs/gust decay, flood monotonicity and wind direction effect, expansion-only, reproducibility, forecast bounds and causality, reliability blend, payload monotonicity and hover recovery. |
| `test_env_and_safety.py` | Observation contract, determinism, valid missions for every profile, action mask, A* path validity, reachability vs real execution, true-future helper does not disturb the env, **no information leakage** to the evaluator, shield legality, V2 weather. |

### 4.8 `outputs/`
| Path | Contents |
|---|---|
| `models/bc_*.pt`, `bc_log.json` | Behaviour-cloned starting policies and their accuracy/success per DAgger round. |
| `models/ppo_*/best.pt / final.pt / last.pt` | Trained PPO policies (best = highest rolling training success after the first third). |
| `models/ppo_*/config.json, progress.csv` | Exact training settings and per-update metrics. |
| `logs/*.log`, `episodes_previous.csv` | Raw console logs of training/tuning/evaluation; results of the evaluation before the deliverability improvements. |
| `paper_build/` | Scratch area for building paper figures and the LaTeX verification copy. Safe to delete. |

### 4.9 `reports/`
| Path | Contents |
|---|---|
| `evaluation/evaluation_report.md` | Main results (headline, all methods, per profile, paired comparisons, threshold tuning). |
| `evaluation/profile_report.md` | Every method × every profile, 14 metrics each, + discussion. |
| `evaluation/episodes.csv` | One row per test episode (9,900 rows) — the source of every number. |
| `evaluation/summary*.csv, results.json` | Aggregated tables. |
| `evaluation/delta.json, delta_tuning.csv` | Chosen thresholds δ = 0.3, δ_abort = 0.85 and the full validation grid. |
| `weather_validation/*` | Validation report, `summary.json`, figures. |
| `training/training_curves.png` | Training curves. |

### 4.10 `IEEE Report/`
| File | Purpose |
|---|---|
| `access.tex` | The filled paper. |
| `figures/` | 12 figures (10 used), PDF + PNG, white background. |
| `paper_numbers.json` | Every number quoted in the paper. |
| `access_preview.pdf` | Preview build (XeTeX); final build should use pdfLaTeX/Overleaf. |
| `access_template.tex/.pdf` | The original IEEE template, kept for reference. |
| `ieeeaccess.cls`, `IEEEtran.cls`, logos, `a1–a3.png`, `fig1.png`, `bullet.png` | Template support files. |
| `BUILD.md` | How to compile and what is still to fill in (authors, bios). |

---

## 5. How the modules work together (one decision, step by step)

1. **Simulator state.** `RescueRouteEnv` holds the true world: wind (`wind.py`), rain + flood (`flood.py`), payload (`stability.py`), battery, time.
2. **Observation.** `_get_obs()` cuts a 17×17 window around the drone (7 layers: walls, no-fly, rooftops, drone, hospital, current water, forecast water) + 15 numbers (battery, deadline, direction to goal, wind, payload…).
3. **Policy.** `PolicyRunner` feeds this into the CNN+LSTM network (`networks.py`) → probability for each of 9 actions (static-geometry moves masked out).
4. **Planner view.** `env.get_planner_view()` gives the shield a copy of only the permitted current information.
5. **Immediate check.** `immediate.py` removes actions that are unsafe right now.
6. **Imagine futures.** `evaluator.py` extends and recalibrates the forecast, draws K = 16 flood futures (and wind futures), and for each candidate action solves "can I still deliver on time, fix the payload, and land afterwards?" exactly in every future → **P_unrec(action)**.
7. **Decide.** `shield.py`: pick the policy's favourite action with P_unrec ≤ 0.3; if none, re-check with 48 futures; if the best action is still below 0.85, pick the policy's favourite among nearly-least-risky actions; otherwise fallback (return to base / rooftop landing / hover).
8. **Act.** `env.step(action)` (or `declare_landing()` for fallback) advances the true world; repeat.
9. **Score afterwards.** `runner.py` records the outcome and checks, using the true future, whether recovery after delivery was really possible.

---

## 6. The algorithms, and why one beats another

### 6.1 Simulating the environment

| Choice | Alternatives | Why this one |
|---|---|---|
| **Vectorised NumPy flood/forecast** | Original per-cell Python loops | Same results bit-for-bit (tested), 1.5× faster; RL needs millions of steps. |
| **Rebalanced flood ranges (v1.1)** | Original ranges | With V1 ranges a clairvoyant planner could finish only 62 % of normal and 0 % of dynamic-flood missions — a benchmark where nothing is possible measures nothing. Only the scenario ranges changed, not the equations. |
| **Training reward v1.1** (hazards −60, geodesic progress) | V1 reward (flood −30, deadline −50, straight-line progress) | V1 made crashing *cheaper* than missing the deadline, so the agent learned to crash on purpose (65 % of episodes). Straight-line progress punished every detour around water. |

### 6.2 Weather models (validated on ERA5, Mumbai monsoon)

**Rain — score on 14 held-out statistics**

| Model | Idea | Score | Verdict |
|---|---|---|---|
| V1 clipped Gaussian AR(1) | Normal noise with memory, clipped to [0,1] | 5/14 | Short wet spells, thin heavy-rain tail, memory dies within hours. |
| Markov chain + Gamma | Wet/dry switches like a coin with memory; amounts random | 6/14 | Gets wet/dry right but every hour's amount is independent → no intensity persistence (lag-1 ACF 0.07 vs 0.91). |
| Latent-Gaussian (one scale) | One hidden Gaussian process; rain = 0 below a threshold, else mapped to Gamma | 8/14 | Persistence of both occurrence and intensity, but memory too short. |
| **Two-scale latent-Gaussian** | Two hidden processes: fast (hours) + slow (days, monsoon active/break spells) | **14/14** | Recovers day-scale memory (lag-24 ACF 0.41 vs 0.49 observed). |

**Wind — score on 18 held-out statistics**

| Model | Score | Verdict |
|---|---|---|
| V1 (AR(1) speed + random-walk direction + rare gusts) | 12/18 | Speed is fine; direction has no prevailing wind (steadiness 0.10 vs 0.82); gusts behave wrongly. |
| VAR(1) on wind vector (u, v) + gust excess | 17/18 | Captures prevailing monsoon direction and persistent gusts; no angle wrap-around problems. |
| **VAR(1) + daily cycle** | 17/18 | Also improves the 24-hour pattern (0.60 vs 0.53, observed 0.82). |

Why the vector model wins: modelling direction as an angle random walk ignores that real wind has a dominant
direction; modelling the east/north components keeps that mean and handles 359°→1° correctly.

### 6.3 Path planning / search

| Algorithm | Where used | Why |
|---|---|---|
| **A\*** (octile heuristic) | Training teacher only | Optimal shortest path on a grid, fast; but it needs the full map and ignores the future, so it is a teacher, not the method. |
| **Dijkstra (geodesic field)** | Training reward | Gives the true "distance around obstacles and water" from every cell in one pass (C-speed via SciPy). |
| **Time-expanded reachability / min-energy dynamic programming** | Shield evaluator, oracle | Floods change over time, so the question is "where can I be at time t?". A layered search over (cell, time) answers it exactly and runs for all actions × all futures at once with array operations. A* on a static map cannot represent water arriving later. |

### 6.4 Learning the pilot

| Approach | Result | Why |
|---|---|---|
| PPO from scratch | ≈0 % success | Long detours never discovered; agent learned to hover until the deadline (hovering also repairs the payload). |
| **Behaviour cloning** (copy the teacher) | 29.3 % test success | Learns quickly from 82,211 teacher decisions, but small mistakes compound because it never saw its own errors. |
| **+ DAgger** | 97 % action agreement | The student flies, the teacher labels where the student actually goes → fixes compounding errors. |
| **+ PPO fine-tuning with critic warm-up and KL anchor** | 33–37 % (+4.1 pp over the clone) | RL optimises the real objective; the warm-up stops a random critic from wrecking the cloned network (without it success fell 37 % → 4 %), and the decaying KL anchor prevents forgetting. |

PPO itself: an actor-critic method that improves the policy using advantages (GAE) but **clips** each update so
the new policy cannot move too far from the old one — stable, simple, standard for discrete actions.

**Network ablation**

| Variant | Test success | UAV lost | Comment |
|---|---|---|---|
| MLP (flattened input) | 34.8 % | 30.0 % | Treats the map as a list of numbers; no notion of spatial neighbourhood. |
| **CNN** | **36.8 %** | 31.2 % | Convolutions see local shapes (walls, water edges) — best raw policy here. |
| CNN + LSTM | 33.4 % | 35.5 % | Memory of past observations; helped less than expected because the forecast channel already carries "where water is heading" and missions are short. |
| CNN + LSTM + risk reward | 35.0 % | 32.8 % | Penalising forecast-risky cells during training lowers losses slightly (−2.7 pp). |

### 6.5 Safety layers (the core comparison, 1,100 paired test missions each)

| Layer | Success | Success + recovery | UAV lost | Future failure | How it decides |
|---|---|---|---|---|---|
| None (raw CNN+LSTM) | 33.4 % | 31.9 % | 35.5 % | 20.8 % | Trusts the policy. |
| **Immediate filter** | 38.4 % | 36.4 % | 31.1 % | 27.1 % | Blocks moves that are unsafe *right now*. |
| **Deterministic recoverability** | 36.5 % | 36.1 % | 6.5 % | 9.0 % | Checks recoverability in **one** "most likely" future. |
| **Probabilistic recoverability (full)** | **38.8 %** | **38.5 %** | **6.1 %** | **7.9 %** | Checks recoverability in **16 sampled** futures (48 to confirm). |

Why each is better than the previous:
- *Immediate filter vs none:* removes obvious crashes (+5 pp success) but cannot see that a safe-now move leads into a corridor that floods later — "future failures" actually *increase* (27.1 %).
- *Recoverability vs immediate:* looks ahead to the end of the mission, including deadline, payload repair time and the flight home → UAV losses drop by 25 pp.
- *Probabilistic vs deterministic:* one "average" future hides uncertainty (a cell with 40 % flood chance is treated as safe); sampling futures sees the spread, giving +2.4 pp success at the same loss rate.

Supporting techniques inside the shield:
- **Threshold coupling of flood samples** — one random field per future, cell floods when its random value drops below the forecast probability → each sampled future matches the forecast exactly and water never recedes (like the simulator).
- **Common random numbers** — all actions are judged on the same futures, so differences between actions are not noise.
- **Confirmation sampling (48)** — with 16 samples a false alarm somewhere in a 40-step mission is almost certain; re-checking before any irreversible decision avoids needless aborts.
- **Indifference margin ε = 0.1** — don't override the policy for a difference smaller than Monte-Carlo noise (this stopped zig-zagging).
- **Forecast recalibration (isotonic regression)** — the raw forecast over-predicts flooding 2–4× at short range; a monotone correction learned on training seeds (Brier 0.115 → 0.103) removed false alarms. Raised success from 35.5 % to 38.8 %.
- **Tuned thresholds** δ = 0.3, δ_abort = 0.85 — chosen on validation seeds to maximise recoverable deliveries with ≤ 8 % UAV loss.

### 6.6 Statistics used

| Method | Used for | Why |
|---|---|---|
| Wilson interval | Success-rate confidence intervals | Accurate for proportions near 0 or 1 (unlike the simple ± formula). |
| Paired bootstrap | Method-vs-method differences | Same missions for every method → pairing removes mission difficulty from the comparison and gives tight intervals. |
| Brier score | Forecast quality | Mean squared error of probability forecasts. |
| Isotonic regression (PAV) | Forecast recalibration | Monotone, no assumed shape, cannot overfit into wiggles. |
| Ljung–Box test | Weather-model residuals | Checks whether a model left unexplained time structure. |
| 95 % simulation envelope | Weather validation | Did the real 2023 statistic fall inside what the model produces in 40 random runs? |

---

## 7. Key results to remember

- Full method: **38.8 %** delivered overall (best of all methods), **71 %** in normal conditions, **6.1 %** UAV loss (vs 31.1 % with the standard filter).
- On missions that are possible at all: **51.2 %** delivered. Only 75.8 % of test missions are possible even with perfect foresight; the four disaster profiles allow only 36–52 %.
- Most remaining aborts are correct (fragile payload cannot be fixed in time; battery cannot reach a landing site after delivery).
- These are simulator results; the flood model is synthetic, so they show the method's *relative* benefit, not real-world rates.

---

## 8. Glossary

| Term | Meaning |
|---|---|
| UAV | Unmanned aerial vehicle (drone). |
| POMDP | Decision problem where the agent sees only part of the state. |
| PPO | Proximal Policy Optimization — the RL algorithm. |
| GAE | Generalised Advantage Estimation — how PPO judges whether an action was better than expected. |
| BC / DAgger | Behaviour cloning (copy a teacher) / Dataset Aggregation (teacher corrects the student's own mistakes). |
| CNN / LSTM / MLP | Convolutional net (spatial) / long short-term memory (temporal) / plain multilayer network. |
| Recoverability | After this action, can the drone still deliver on time with a valid payload and then land safely? |
| P_unrec | Fraction of sampled futures in which the answer is "no". |
| δ, δ_abort, ε | Safety threshold (0.3), abort threshold (0.85), indifference margin (0.1). |
| Oracle-feasible | A planner that knows the true future flood could deliver on time. |
| ERA5 | Global hourly weather reanalysis dataset (Copernicus/ECMWF). |
| V1 / V1.1 / V2 | Original environment / rebalanced flood ranges + fixes / ERA5-calibrated weather option. |
