# RescueRoute — Detailed results per stress profile and method

*Generated from `episodes.csv` (9,900 test episodes: 9 methods × 11 profiles × 100 held-out seeds 2,000,000+). Every method flew the same missions with the same hidden weather and flood. Shield thresholds δ = 0.3, δ_abort = 0.85 (tuned on validation seeds only).*

## Contents

1. Metric definitions
2. Stress profiles
3. Overview matrices (success, UAV lost, safe abort)
4. Detailed table for every method
5. Full method vs immediate safety filter, per profile (paired)
6. Discussion: why the pooled success is ~39 %, training/step-size effects, real-world meaning

## 1. Metric definitions

| Metric | Meaning |
|---|---|
| Success | Delivered on time with a valid payload (95 % Wilson CI in brackets). |
| Success+rec. | Success, and on the TRUE future flood the UAV could still reach a dry base/rooftop with ≥ 10 % battery reserve. |
| UAV lost | Crash, flood hazard/engulfment, or empty battery. |
| Flood / Collision / Deadline / Payload / Battery | How the failed missions ended. |
| Safe abort | The fallback controller returned to base or landed on a rooftop (mission abandoned, UAV saved). |
| Future failure | Mission failed although every executed action had passed the immediate (one-step) safety check. |
| Interv. | Fraction of decisions where the safety layer overrode the policy. |
| Oracle-feasible | A planner that knows the entire true future flood could deliver on time (ignores payload and return flight, so it is an UPPER bound). |
| Succ. if feasible | Success rate restricted to oracle-feasible missions. |
| Steps / Energy | Mean mission length and energy used, over successful missions. |

## 2. Stress profiles

| Profile | Definition | Oracle-feasible missions |
|---|---|---:|
| normal | Moderate wind/rain, normal battery, relaxed deadline. | 95.0% |
| low battery | Battery ~1.3-1.9x the straight-line distance: delivery possible, return marginal. | 95.0% |
| tight deadline | Deadline only 1.12-1.30x the minimum steps. | 94.0% |
| strong wind | Mean wind 11-14 m/s, high variability, frequent gusts. | 93.0% |
| dynamic flood | Heavy rain, twice the initial flooding, 1.5x spread rate. | 52.0% |
| fragile | Every payload FRAGILE (must arrive with stability >= 0.80). | 95.0% |
| wind flood | Strong wind + dynamic flooding. | 39.0% |
| low battery deadline | Low battery + tight deadline. | 94.0% |
| compound | Low battery + tight deadline + wind + flood. | 36.0% |
| all combined | Every stress factor at once, fragile payload included. | 46.0% |
| low forecast reliability | Forecast reliability alpha in [0.20, 0.45]. | 95.0% |
| **all profiles** | | **75.8%** |

## 3. Overview matrices

### Mission success

| Method | normal | low battery | tight deadline | strong wind | dynamic flood | fragile | wind flood | low battery deadline | compound | all combined | low forecast reliability | **All** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Random | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | **0.0%** |
| BC student (no PPO) | 55.0% | 50.0% | 42.0% | 35.0% | 8.0% | 32.0% | 1.0% | 43.0% | 1.0% | 0.0% | 55.0% | **29.3%** |
| PPO (MLP) | 65.0% | 61.0% | 51.0% | 34.0% | 8.0% | 44.0% | 2.0% | 53.0% | 1.0% | 0.0% | 64.0% | **34.8%** |
| PPO+CNN | 68.0% | 66.0% | 46.0% | 39.0% | 5.0% | 55.0% | 4.0% | 54.0% | 2.0% | 0.0% | 66.0% | **36.8%** |
| PPO+CNN+LSTM | 60.0% | 56.0% | 50.0% | 37.0% | 8.0% | 43.0% | 7.0% | 45.0% | 3.0% | 0.0% | 58.0% | **33.4%** |
| PPO+CNN+LSTM+Risk | 66.0% | 56.0% | 51.0% | 36.0% | 8.0% | 49.0% | 6.0% | 47.0% | 5.0% | 0.0% | 61.0% | **35.0%** |
| +Immediate safety | 69.0% | 63.0% | 60.0% | 42.0% | 10.0% | 49.0% | 7.0% | 50.0% | 4.0% | 0.0% | 68.0% | **38.4%** |
| +Deterministic recov. | 69.0% | 54.0% | 65.0% | 41.0% | 10.0% | 48.0% | 3.0% | 41.0% | 1.0% | 0.0% | 69.0% | **36.5%** |
| **+Probabilistic recov. (full)** | 71.0% | 60.0% | 64.0% | 45.0% | 13.0% | 53.0% | 5.0% | 44.0% | 3.0% | 0.0% | 69.0% | **38.8%** |

### UAV lost

Lower is better.

| Method | normal | low battery | tight deadline | strong wind | dynamic flood | fragile | wind flood | low battery deadline | compound | all combined | low forecast reliability | **All** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Random | 99.0% | 100.0% | 93.0% | 98.0% | 100.0% | 99.0% | 100.0% | 94.0% | 100.0% | 100.0% | 99.0% | **98.4%** |
| BC student (no PPO) | 24.0% | 33.0% | 15.0% | 25.0% | 65.0% | 30.0% | 76.0% | 18.0% | 51.0% | 56.0% | 22.0% | **37.7%** |
| PPO (MLP) | 15.0% | 22.0% | 13.0% | 18.0% | 57.0% | 18.0% | 61.0% | 8.0% | 57.0% | 45.0% | 16.0% | **30.0%** |
| PPO+CNN | 20.0% | 24.0% | 10.0% | 22.0% | 57.0% | 27.0% | 67.0% | 13.0% | 34.0% | 51.0% | 18.0% | **31.2%** |
| PPO+CNN+LSTM | 23.0% | 30.0% | 15.0% | 28.0% | 67.0% | 26.0% | 66.0% | 17.0% | 40.0% | 53.0% | 26.0% | **35.5%** |
| PPO+CNN+LSTM+Risk | 18.0% | 25.0% | 13.0% | 29.0% | 68.0% | 18.0% | 63.0% | 15.0% | 42.0% | 54.0% | 16.0% | **32.8%** |
| +Immediate safety | 13.0% | 21.0% | 9.0% | 21.0% | 66.0% | 17.0% | 67.0% | 11.0% | 47.0% | 53.0% | 17.0% | **31.1%** |
| +Deterministic recov. | 7.0% | 3.0% | 6.0% | 13.0% | 11.0% | 9.0% | 6.0% | 3.0% | 5.0% | 1.0% | 7.0% | **6.5%** |
| **+Probabilistic recov. (full)** | 7.0% | 3.0% | 4.0% | 9.0% | 13.0% | 9.0% | 5.0% | 2.0% | 4.0% | 0.0% | 11.0% | **6.1%** |

### Safe abort

Only the recoverability shields can abort; every other method keeps flying until it delivers, crashes or times out.

| Method | normal | low battery | tight deadline | strong wind | dynamic flood | fragile | wind flood | low battery deadline | compound | all combined | low forecast reliability | **All** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Random | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | **0.0%** |
| BC student (no PPO) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | **0.0%** |
| PPO (MLP) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | **0.0%** |
| PPO+CNN | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | **0.0%** |
| PPO+CNN+LSTM | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | **0.0%** |
| PPO+CNN+LSTM+Risk | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | **0.0%** |
| +Immediate safety | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | **0.0%** |
| +Deterministic recov. | 14.0% | 33.0% | 22.0% | 34.0% | 74.0% | 26.0% | 90.0% | 37.0% | 90.0% | 99.0% | 15.0% | **48.5%** |
| **+Probabilistic recov. (full)** | 15.0% | 32.0% | 28.0% | 40.0% | 72.0% | 28.0% | 90.0% | 50.0% | 92.0% | 100.0% | 13.0% | **50.9%** |

### Success on oracle-feasible missions only

Removes missions that no controller could have completed.

| Method | normal | low battery | tight deadline | strong wind | dynamic flood | fragile | wind flood | low battery deadline | compound | all combined | low forecast reliability | **All** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Random | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | **0.0%** |
| BC student (no PPO) | 57.9% | 52.6% | 44.7% | 37.6% | 15.4% | 33.7% | 2.6% | 45.7% | 2.8% | 0.0% | 57.9% | **38.6%** |
| PPO (MLP) | 68.4% | 64.2% | 54.3% | 36.6% | 15.4% | 46.3% | 5.1% | 56.4% | 2.8% | 0.0% | 67.4% | **45.9%** |
| PPO+CNN | 71.6% | 69.5% | 48.9% | 41.9% | 9.6% | 57.9% | 10.3% | 57.4% | 5.6% | 0.0% | 69.5% | **48.6%** |
| PPO+CNN+LSTM | 63.2% | 58.9% | 53.2% | 39.8% | 15.4% | 45.3% | 17.9% | 47.9% | 8.3% | 0.0% | 61.1% | **44.0%** |
| PPO+CNN+LSTM+Risk | 69.5% | 58.9% | 54.3% | 38.7% | 15.4% | 51.6% | 15.4% | 50.0% | 13.9% | 0.0% | 64.2% | **46.2%** |
| +Immediate safety | 72.6% | 66.3% | 63.8% | 45.2% | 19.2% | 51.6% | 17.9% | 53.2% | 11.1% | 0.0% | 71.6% | **50.6%** |
| +Deterministic recov. | 72.6% | 56.8% | 69.1% | 44.1% | 19.2% | 50.5% | 7.7% | 43.6% | 2.8% | 0.0% | 72.6% | **48.1%** |
| **+Probabilistic recov. (full)** | 74.7% | 63.2% | 68.1% | 48.4% | 25.0% | 55.8% | 12.8% | 46.8% | 8.3% | 0.0% | 72.6% | **51.2%** |

## 4. Detailed table for every method

### Random

*Uniform random actions (sanity floor)*

| Profile | Success (95% CI) | Success+rec. | UAV lost | Flood | Collision | Deadline | Payload | Battery | Safe abort | Future failure | Interv. | Succ. if feasible | Steps | Energy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| normal | 0.0% (0.0%–3.7%) | 0.0% | 99.0% | 39.0% | 60.0% | 1.0% | 0.0% | 0.0% | 0.0% | 2.0% | 0.0% | 0.0% | – | – |
| low battery | 0.0% (0.0%–3.7%) | 0.0% | 100.0% | 37.0% | 57.0% | 0.0% | 0.0% | 6.0% | 0.0% | 2.0% | 0.0% | 0.0% | – | – |
| tight deadline | 0.0% (0.0%–3.7%) | 0.0% | 93.0% | 36.0% | 57.0% | 7.0% | 0.0% | 0.0% | 0.0% | 2.0% | 0.0% | 0.0% | – | – |
| strong wind | 0.0% (0.0%–3.7%) | 0.0% | 98.0% | 40.0% | 58.0% | 2.0% | 0.0% | 0.0% | 0.0% | 2.0% | 0.0% | 0.0% | – | – |
| dynamic flood | 0.0% (0.0%–3.7%) | 0.0% | 100.0% | 59.0% | 41.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | – | – |
| fragile | 0.0% (0.0%–3.7%) | 0.0% | 99.0% | 39.0% | 60.0% | 1.0% | 0.0% | 0.0% | 0.0% | 2.0% | 0.0% | 0.0% | – | – |
| wind flood | 0.0% (0.0%–3.7%) | 0.0% | 100.0% | 59.0% | 41.0% | 0.0% | 0.0% | 0.0% | 0.0% | 3.0% | 0.0% | 0.0% | – | – |
| low battery deadline | 0.0% (0.0%–3.7%) | 0.0% | 94.0% | 36.0% | 57.0% | 6.0% | 0.0% | 1.0% | 0.0% | 2.0% | 0.0% | 0.0% | – | – |
| compound | 0.0% (0.0%–3.7%) | 0.0% | 100.0% | 62.0% | 38.0% | 0.0% | 0.0% | 0.0% | 0.0% | 5.0% | 0.0% | 0.0% | – | – |
| all combined | 0.0% (0.0%–3.7%) | 0.0% | 100.0% | 61.0% | 39.0% | 0.0% | 0.0% | 0.0% | 0.0% | 5.0% | 0.0% | 0.0% | – | – |
| low forecast reliability | 0.0% (0.0%–3.7%) | 0.0% | 99.0% | 39.0% | 60.0% | 1.0% | 0.0% | 0.0% | 0.0% | 2.0% | 0.0% | 0.0% | – | – |
| **All profiles** | 0.0% (0.0%–0.3%) | 0.0% | 98.4% | 46.1% | 51.6% | 1.6% | 0.0% | 0.6% | 0.0% | 2.5% | 0.0% | 0.0% | – | – |

Best profile: **normal** (0.0%); hardest: **normal** (0.0%).

### BC student (no PPO)

*Behaviour-cloned CNN+LSTM student before PPO (initialisation only)*

| Profile | Success (95% CI) | Success+rec. | UAV lost | Flood | Collision | Deadline | Payload | Battery | Safe abort | Future failure | Interv. | Succ. if feasible | Steps | Energy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| normal | 55.0% (45.2%–64.4%) | 55.0% | 24.0% | 24.0% | 0.0% | 14.0% | 7.0% | 0.0% | 0.0% | 17.0% | 0.0% | 57.9% | 33.3 | 38.0 |
| low battery | 50.0% (40.4%–59.6%) | 42.0% | 33.0% | 26.0% | 0.0% | 6.0% | 11.0% | 7.0% | 0.0% | 22.0% | 0.0% | 52.6% | 32.2 | 37.4 |
| tight deadline | 42.0% (32.8%–51.8%) | 42.0% | 15.0% | 15.0% | 0.0% | 39.0% | 4.0% | 0.0% | 0.0% | 10.0% | 0.0% | 44.7% | 31.0 | 37.2 |
| strong wind | 35.0% (26.4%–44.7%) | 35.0% | 25.0% | 25.0% | 0.0% | 30.0% | 10.0% | 0.0% | 0.0% | 24.0% | 0.0% | 37.6% | 29.9 | 36.5 |
| dynamic flood | 8.0% (4.1%–15.0%) | 8.0% | 65.0% | 65.0% | 0.0% | 27.0% | 0.0% | 0.0% | 0.0% | 39.0% | 0.0% | 15.4% | 31.8 | 35.5 |
| fragile | 32.0% (23.7%–41.7%) | 32.0% | 30.0% | 30.0% | 0.0% | 27.0% | 11.0% | 0.0% | 0.0% | 27.0% | 0.0% | 33.7% | 41.5 | 41.7 |
| wind flood | 1.0% (0.2%–5.4%) | 1.0% | 76.0% | 76.0% | 0.0% | 23.0% | 0.0% | 0.0% | 0.0% | 43.0% | 0.0% | 2.6% | 30.0 | 40.2 |
| low battery deadline | 43.0% (33.7%–52.8%) | 36.0% | 18.0% | 16.0% | 0.0% | 33.0% | 6.0% | 2.0% | 0.0% | 11.0% | 0.0% | 45.7% | 30.9 | 37.1 |
| compound | 1.0% (0.2%–5.4%) | 1.0% | 51.0% | 51.0% | 0.0% | 48.0% | 0.0% | 0.0% | 0.0% | 18.0% | 0.0% | 2.8% | 28.0 | 29.7 |
| all combined | 0.0% (0.0%–3.7%) | 0.0% | 56.0% | 56.0% | 0.0% | 43.0% | 1.0% | 0.0% | 0.0% | 25.0% | 0.0% | 0.0% | – | – |
| low forecast reliability | 55.0% (45.2%–64.4%) | 55.0% | 22.0% | 22.0% | 0.0% | 14.0% | 9.0% | 0.0% | 0.0% | 22.0% | 0.0% | 57.9% | 32.7 | 37.8 |
| **All profiles** | 29.3% (26.7%–32.0%) | 27.9% | 37.7% | 36.9% | 0.0% | 27.6% | 5.4% | 0.8% | 0.0% | 23.5% | 0.0% | 38.6% | 32.8 | 37.8 |

Best profile: **normal** (55.0%); hardest: **all combined** (0.0%).

### PPO (MLP)

*PPO, flattened observation (MLP)*

| Profile | Success (95% CI) | Success+rec. | UAV lost | Flood | Collision | Deadline | Payload | Battery | Safe abort | Future failure | Interv. | Succ. if feasible | Steps | Energy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| normal | 65.0% (55.3%–73.6%) | 65.0% | 15.0% | 15.0% | 0.0% | 12.0% | 8.0% | 0.0% | 0.0% | 20.0% | 0.0% | 68.4% | 34.2 | 38.3 |
| low battery | 61.0% (51.2%–70.0%) | 52.0% | 22.0% | 14.0% | 0.0% | 6.0% | 11.0% | 8.0% | 0.0% | 21.0% | 0.0% | 64.2% | 32.8 | 37.7 |
| tight deadline | 51.0% (41.3%–60.6%) | 51.0% | 13.0% | 13.0% | 0.0% | 34.0% | 2.0% | 0.0% | 0.0% | 12.0% | 0.0% | 54.3% | 30.9 | 36.7 |
| strong wind | 34.0% (25.5%–43.7%) | 34.0% | 18.0% | 18.0% | 0.0% | 39.0% | 9.0% | 0.0% | 0.0% | 24.0% | 0.0% | 36.6% | 30.0 | 36.2 |
| dynamic flood | 8.0% (4.1%–15.0%) | 7.0% | 57.0% | 57.0% | 0.0% | 35.0% | 0.0% | 0.0% | 0.0% | 42.0% | 0.0% | 15.4% | 32.4 | 37.4 |
| fragile | 44.0% (34.7%–53.8%) | 44.0% | 18.0% | 18.0% | 0.0% | 24.0% | 14.0% | 0.0% | 0.0% | 27.0% | 0.0% | 46.3% | 41.9 | 41.6 |
| wind flood | 2.0% (0.6%–7.0%) | 2.0% | 61.0% | 61.0% | 0.0% | 36.0% | 1.0% | 0.0% | 0.0% | 42.0% | 0.0% | 5.1% | 30.0 | 33.9 |
| low battery deadline | 53.0% (43.3%–62.5%) | 37.0% | 8.0% | 7.0% | 0.0% | 31.0% | 8.0% | 1.0% | 0.0% | 14.0% | 0.0% | 56.4% | 30.7 | 36.8 |
| compound | 1.0% (0.2%–5.4%) | 0.0% | 57.0% | 54.0% | 0.0% | 42.0% | 0.0% | 3.0% | 0.0% | 21.0% | 0.0% | 2.8% | 34.0 | 41.8 |
| all combined | 0.0% (0.0%–3.7%) | 0.0% | 45.0% | 44.0% | 0.0% | 55.0% | 0.0% | 1.0% | 0.0% | 20.0% | 0.0% | 0.0% | – | – |
| low forecast reliability | 64.0% (54.2%–72.7%) | 64.0% | 16.0% | 16.0% | 0.0% | 15.0% | 5.0% | 0.0% | 0.0% | 17.0% | 0.0% | 67.4% | 34.5 | 38.4 |
| **All profiles** | 34.8% (32.1%–37.7%) | 32.4% | 30.0% | 28.8% | 0.0% | 29.9% | 5.3% | 1.2% | 0.0% | 23.6% | 0.0% | 45.9% | 33.5 | 38.0 |

Best profile: **normal** (65.0%); hardest: **all combined** (0.0%).

### PPO+CNN

*PPO + CNN encoder*

| Profile | Success (95% CI) | Success+rec. | UAV lost | Flood | Collision | Deadline | Payload | Battery | Safe abort | Future failure | Interv. | Succ. if feasible | Steps | Energy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| normal | 68.0% (58.3%–76.3%) | 68.0% | 20.0% | 20.0% | 0.0% | 12.0% | 0.0% | 0.0% | 0.0% | 12.0% | 0.0% | 71.6% | 34.5 | 38.3 |
| low battery | 66.0% (56.3%–74.5%) | 51.0% | 24.0% | 18.0% | 0.0% | 8.0% | 2.0% | 6.0% | 0.0% | 11.0% | 0.0% | 69.5% | 33.9 | 38.1 |
| tight deadline | 46.0% (36.6%–55.7%) | 46.0% | 10.0% | 10.0% | 0.0% | 38.0% | 6.0% | 0.0% | 0.0% | 11.0% | 0.0% | 48.9% | 31.1 | 36.6 |
| strong wind | 39.0% (30.0%–48.8%) | 39.0% | 22.0% | 22.0% | 0.0% | 29.0% | 10.0% | 0.0% | 0.0% | 24.0% | 0.0% | 41.9% | 31.8 | 37.6 |
| dynamic flood | 5.0% (2.2%–11.2%) | 5.0% | 57.0% | 57.0% | 0.0% | 38.0% | 0.0% | 0.0% | 0.0% | 43.0% | 0.0% | 9.6% | 31.4 | 34.7 |
| fragile | 55.0% (45.2%–64.4%) | 55.0% | 27.0% | 27.0% | 0.0% | 17.0% | 1.0% | 0.0% | 0.0% | 18.0% | 0.0% | 57.9% | 42.6 | 42.3 |
| wind flood | 4.0% (1.6%–9.8%) | 4.0% | 67.0% | 67.0% | 0.0% | 29.0% | 0.0% | 0.0% | 0.0% | 45.0% | 0.0% | 10.3% | 34.2 | 39.2 |
| low battery deadline | 54.0% (44.3%–63.4%) | 41.0% | 13.0% | 13.0% | 0.0% | 27.0% | 6.0% | 0.0% | 0.0% | 13.0% | 0.0% | 57.4% | 31.6 | 37.0 |
| compound | 2.0% (0.6%–7.0%) | 2.0% | 34.0% | 34.0% | 0.0% | 64.0% | 0.0% | 0.0% | 0.0% | 16.0% | 0.0% | 5.6% | 31.5 | 34.5 |
| all combined | 0.0% (0.0%–3.7%) | 0.0% | 51.0% | 51.0% | 0.0% | 49.0% | 0.0% | 0.0% | 0.0% | 23.0% | 0.0% | 0.0% | – | – |
| low forecast reliability | 66.0% (56.3%–74.5%) | 66.0% | 18.0% | 18.0% | 0.0% | 13.0% | 3.0% | 0.0% | 0.0% | 15.0% | 0.0% | 69.5% | 34.1 | 38.2 |
| **All profiles** | 36.8% (34.0%–39.7%) | 34.3% | 31.2% | 30.6% | 0.0% | 29.5% | 2.5% | 0.5% | 0.0% | 21.0% | 0.0% | 48.6% | 34.3 | 38.3 |

Best profile: **normal** (68.0%); hardest: **all combined** (0.0%).

### PPO+CNN+LSTM

*PPO + CNN + LSTM*

| Profile | Success (95% CI) | Success+rec. | UAV lost | Flood | Collision | Deadline | Payload | Battery | Safe abort | Future failure | Interv. | Succ. if feasible | Steps | Energy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| normal | 60.0% (50.2%–69.1%) | 60.0% | 23.0% | 23.0% | 0.0% | 14.0% | 3.0% | 0.0% | 0.0% | 10.0% | 0.0% | 63.2% | 33.6 | 37.9 |
| low battery | 56.0% (46.2%–65.3%) | 49.0% | 30.0% | 22.0% | 0.0% | 7.0% | 7.0% | 8.0% | 0.0% | 13.0% | 0.0% | 58.9% | 32.8 | 37.5 |
| tight deadline | 50.0% (40.4%–59.6%) | 50.0% | 15.0% | 15.0% | 0.0% | 32.0% | 3.0% | 0.0% | 0.0% | 7.0% | 0.0% | 53.2% | 31.3 | 37.3 |
| strong wind | 37.0% (28.2%–46.8%) | 37.0% | 28.0% | 28.0% | 0.0% | 28.0% | 7.0% | 0.0% | 0.0% | 19.0% | 0.0% | 39.8% | 31.9 | 37.8 |
| dynamic flood | 8.0% (4.1%–15.0%) | 8.0% | 67.0% | 67.0% | 0.0% | 25.0% | 0.0% | 0.0% | 0.0% | 50.0% | 0.0% | 15.4% | 32.4 | 35.5 |
| fragile | 43.0% (33.7%–52.8%) | 43.0% | 26.0% | 26.0% | 0.0% | 25.0% | 6.0% | 0.0% | 0.0% | 17.0% | 0.0% | 45.3% | 42.2 | 42.0 |
| wind flood | 7.0% (3.4%–13.7%) | 7.0% | 66.0% | 66.0% | 0.0% | 27.0% | 0.0% | 0.0% | 0.0% | 45.0% | 0.0% | 17.9% | 30.6 | 35.9 |
| low battery deadline | 45.0% (35.6%–54.8%) | 37.0% | 17.0% | 17.0% | 0.0% | 27.0% | 11.0% | 0.0% | 0.0% | 15.0% | 0.0% | 47.9% | 30.6 | 36.9 |
| compound | 3.0% (1.0%–8.5%) | 2.0% | 40.0% | 39.0% | 0.0% | 57.0% | 0.0% | 1.0% | 0.0% | 14.0% | 0.0% | 8.3% | 28.7 | 31.5 |
| all combined | 0.0% (0.0%–3.7%) | 0.0% | 53.0% | 53.0% | 0.0% | 47.0% | 0.0% | 0.0% | 0.0% | 25.0% | 0.0% | 0.0% | – | – |
| low forecast reliability | 58.0% (48.2%–67.2%) | 58.0% | 26.0% | 26.0% | 0.0% | 12.0% | 4.0% | 0.0% | 0.0% | 14.0% | 0.0% | 61.1% | 34.1 | 38.2 |
| **All profiles** | 33.4% (30.6%–36.2%) | 31.9% | 35.5% | 34.7% | 0.0% | 27.4% | 3.7% | 0.8% | 0.0% | 20.8% | 0.0% | 44.0% | 33.6 | 38.0 |

Best profile: **normal** (60.0%); hardest: **all combined** (0.0%).

### PPO+CNN+LSTM+Risk

*PPO + CNN + LSTM trained with a forecast-risk penalty*

| Profile | Success (95% CI) | Success+rec. | UAV lost | Flood | Collision | Deadline | Payload | Battery | Safe abort | Future failure | Interv. | Succ. if feasible | Steps | Energy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| normal | 66.0% (56.3%–74.5%) | 66.0% | 18.0% | 18.0% | 0.0% | 14.0% | 2.0% | 0.0% | 0.0% | 5.0% | 0.0% | 69.5% | 35.7 | 39.3 |
| low battery | 56.0% (46.2%–65.3%) | 49.0% | 25.0% | 17.0% | 0.0% | 14.0% | 5.0% | 8.0% | 0.0% | 8.0% | 0.0% | 58.9% | 33.2 | 38.1 |
| tight deadline | 51.0% (41.3%–60.6%) | 51.0% | 13.0% | 13.0% | 0.0% | 33.0% | 3.0% | 0.0% | 0.0% | 5.0% | 0.0% | 54.3% | 31.6 | 37.4 |
| strong wind | 36.0% (27.3%–45.8%) | 36.0% | 29.0% | 29.0% | 0.0% | 30.0% | 5.0% | 0.0% | 0.0% | 19.0% | 0.0% | 38.7% | 30.7 | 36.7 |
| dynamic flood | 8.0% (4.1%–15.0%) | 8.0% | 68.0% | 68.0% | 0.0% | 24.0% | 0.0% | 0.0% | 0.0% | 44.0% | 0.0% | 15.4% | 31.8 | 36.4 |
| fragile | 49.0% (39.4%–58.7%) | 49.0% | 18.0% | 18.0% | 0.0% | 28.0% | 5.0% | 0.0% | 0.0% | 12.0% | 0.0% | 51.6% | 43.2 | 43.0 |
| wind flood | 6.0% (2.8%–12.5%) | 6.0% | 63.0% | 63.0% | 0.0% | 31.0% | 0.0% | 0.0% | 0.0% | 38.0% | 0.0% | 15.4% | 30.5 | 36.1 |
| low battery deadline | 47.0% (37.5%–56.7%) | 39.0% | 15.0% | 15.0% | 0.0% | 29.0% | 9.0% | 0.0% | 0.0% | 12.0% | 0.0% | 50.0% | 31.3 | 37.6 |
| compound | 5.0% (2.2%–11.2%) | 4.0% | 42.0% | 42.0% | 0.0% | 53.0% | 0.0% | 0.0% | 0.0% | 15.0% | 0.0% | 13.9% | 30.2 | 35.2 |
| all combined | 0.0% (0.0%–3.7%) | 0.0% | 54.0% | 54.0% | 0.0% | 45.0% | 1.0% | 0.0% | 0.0% | 21.0% | 0.0% | 0.0% | – | – |
| low forecast reliability | 61.0% (51.2%–70.0%) | 61.0% | 16.0% | 16.0% | 0.0% | 19.0% | 4.0% | 0.0% | 0.0% | 10.0% | 0.0% | 64.2% | 34.7 | 38.8 |
| **All profiles** | 35.0% (32.2%–37.9%) | 33.5% | 32.8% | 32.1% | 0.0% | 29.1% | 3.1% | 0.7% | 0.0% | 17.2% | 0.0% | 46.2% | 34.4 | 38.6 |

Best profile: **normal** (66.0%); hardest: **all combined** (0.0%).

### +Immediate safety

*PPO + CNN + LSTM + immediate one-step safety filter*

| Profile | Success (95% CI) | Success+rec. | UAV lost | Flood | Collision | Deadline | Payload | Battery | Safe abort | Future failure | Interv. | Succ. if feasible | Steps | Energy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| normal | 69.0% (59.4%–77.2%) | 69.0% | 13.0% | 13.0% | 0.0% | 13.0% | 5.0% | 0.0% | 0.0% | 15.0% | 1.1% | 72.6% | 34.6 | 38.6 |
| low battery | 63.0% (53.2%–71.8%) | 53.0% | 21.0% | 12.0% | 0.0% | 7.0% | 9.0% | 9.0% | 0.0% | 17.0% | 1.1% | 66.3% | 33.4 | 37.7 |
| tight deadline | 60.0% (50.2%–69.1%) | 60.0% | 9.0% | 9.0% | 0.0% | 18.0% | 13.0% | 0.0% | 0.0% | 19.0% | 2.4% | 63.8% | 32.4 | 37.7 |
| strong wind | 42.0% (32.8%–51.8%) | 42.0% | 21.0% | 21.0% | 0.0% | 24.0% | 13.0% | 0.0% | 0.0% | 29.0% | 1.6% | 45.2% | 31.8 | 37.6 |
| dynamic flood | 10.0% (5.5%–17.4%) | 10.0% | 66.0% | 66.0% | 0.0% | 24.0% | 0.0% | 0.0% | 0.0% | 54.0% | 1.6% | 19.2% | 32.4 | 36.3 |
| fragile | 49.0% (39.4%–58.7%) | 49.0% | 17.0% | 17.0% | 0.0% | 25.0% | 9.0% | 0.0% | 0.0% | 22.0% | 1.3% | 51.6% | 42.5 | 42.5 |
| wind flood | 7.0% (3.4%–13.7%) | 7.0% | 67.0% | 67.0% | 0.0% | 26.0% | 0.0% | 0.0% | 0.0% | 49.0% | 1.7% | 17.9% | 30.6 | 35.9 |
| low battery deadline | 50.0% (40.4%–59.6%) | 39.0% | 11.0% | 10.0% | 0.0% | 21.0% | 18.0% | 1.0% | 0.0% | 23.0% | 1.8% | 53.2% | 31.1 | 37.0 |
| compound | 4.0% (1.6%–9.8%) | 3.0% | 47.0% | 47.0% | 0.0% | 49.0% | 0.0% | 0.0% | 0.0% | 21.0% | 4.3% | 11.1% | 31.0 | 33.0 |
| all combined | 0.0% (0.0%–3.7%) | 0.0% | 53.0% | 53.0% | 0.0% | 46.0% | 1.0% | 0.0% | 0.0% | 31.0% | 4.2% | 0.0% | – | – |
| low forecast reliability | 68.0% (58.3%–76.3%) | 68.0% | 17.0% | 17.0% | 0.0% | 9.0% | 6.0% | 0.0% | 0.0% | 18.0% | 1.2% | 71.6% | 35.0 | 38.8 |
| **All profiles** | 38.4% (35.5%–41.3%) | 36.4% | 31.1% | 30.2% | 0.0% | 23.8% | 6.7% | 0.9% | 0.0% | 27.1% | 2.0% | 50.6% | 34.3 | 38.4 |

Best profile: **normal** (69.0%); hardest: **all combined** (0.0%).

### +Deterministic recov.

*PPO + CNN + LSTM + recoverability on one nominal future*

| Profile | Success (95% CI) | Success+rec. | UAV lost | Flood | Collision | Deadline | Payload | Battery | Safe abort | Future failure | Interv. | Succ. if feasible | Steps | Energy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| normal | 69.0% (59.4%–77.2%) | 69.0% | 7.0% | 7.0% | 0.0% | 5.0% | 5.0% | 0.0% | 14.0% | 12.0% | 5.5% | 72.6% | 34.6 | 38.7 |
| low battery | 54.0% (44.3%–63.4%) | 53.0% | 3.0% | 2.0% | 0.0% | 6.0% | 4.0% | 1.0% | 33.0% | 6.0% | 20.4% | 56.8% | 32.7 | 36.8 |
| tight deadline | 65.0% (55.3%–73.6%) | 64.0% | 6.0% | 6.0% | 0.0% | 5.0% | 2.0% | 0.0% | 22.0% | 8.0% | 11.9% | 69.1% | 32.9 | 38.1 |
| strong wind | 41.0% (31.9%–50.8%) | 41.0% | 13.0% | 13.0% | 0.0% | 6.0% | 6.0% | 0.0% | 34.0% | 17.0% | 10.9% | 44.1% | 31.4 | 37.5 |
| dynamic flood | 10.0% (5.5%–17.4%) | 10.0% | 11.0% | 11.0% | 0.0% | 5.0% | 0.0% | 0.0% | 74.0% | 10.0% | 58.3% | 19.2% | 32.4 | 36.3 |
| fragile | 48.0% (38.5%–57.7%) | 48.0% | 9.0% | 9.0% | 0.0% | 9.0% | 8.0% | 0.0% | 26.0% | 17.0% | 7.4% | 50.5% | 42.4 | 42.6 |
| wind flood | 3.0% (1.0%–8.5%) | 3.0% | 6.0% | 6.0% | 0.0% | 1.0% | 0.0% | 0.0% | 90.0% | 6.0% | 69.6% | 7.7% | 29.3 | 34.1 |
| low battery deadline | 41.0% (31.9%–50.8%) | 39.0% | 3.0% | 3.0% | 0.0% | 14.0% | 5.0% | 0.0% | 37.0% | 8.0% | 22.8% | 43.6% | 30.3 | 36.1 |
| compound | 1.0% (0.2%–5.4%) | 1.0% | 5.0% | 5.0% | 0.0% | 4.0% | 0.0% | 0.0% | 90.0% | 3.0% | 79.6% | 2.8% | 28.0 | 28.6 |
| all combined | 0.0% (0.0%–3.7%) | 0.0% | 1.0% | 1.0% | 0.0% | 0.0% | 0.0% | 0.0% | 99.0% | 0.0% | 89.1% | 0.0% | – | – |
| low forecast reliability | 69.0% (59.4%–77.2%) | 69.0% | 7.0% | 7.0% | 0.0% | 4.0% | 5.0% | 0.0% | 15.0% | 12.0% | 6.5% | 72.6% | 35.2 | 39.2 |
| **All profiles** | 36.5% (33.7%–39.3%) | 36.1% | 6.5% | 6.4% | 0.0% | 5.4% | 3.2% | 0.1% | 48.5% | 9.0% | 34.7% | 48.1% | 34.2 | 38.4 |

Best profile: **normal** (69.0%); hardest: **all combined** (0.0%).

### +Probabilistic recov. (full)

*Full method: PPO + CNN + LSTM + Monte-Carlo recoverability shield*

| Profile | Success (95% CI) | Success+rec. | UAV lost | Flood | Collision | Deadline | Payload | Battery | Safe abort | Future failure | Interv. | Succ. if feasible | Steps | Energy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| normal | 71.0% (61.5%–79.0%) | 71.0% | 7.0% | 7.0% | 0.0% | 4.0% | 3.0% | 0.0% | 15.0% | 10.0% | 6.5% | 74.7% | 35.3 | 39.0 |
| low battery | 60.0% (50.2%–69.1%) | 60.0% | 3.0% | 3.0% | 0.0% | 1.0% | 4.0% | 0.0% | 32.0% | 7.0% | 22.8% | 63.2% | 33.4 | 37.6 |
| tight deadline | 64.0% (54.2%–72.7%) | 63.0% | 4.0% | 4.0% | 0.0% | 2.0% | 2.0% | 0.0% | 28.0% | 5.0% | 23.5% | 68.1% | 33.0 | 38.4 |
| strong wind | 45.0% (35.6%–54.8%) | 45.0% | 9.0% | 9.0% | 0.0% | 2.0% | 4.0% | 0.0% | 40.0% | 13.0% | 22.5% | 48.4% | 33.9 | 38.7 |
| dynamic flood | 13.0% (7.8%–21.0%) | 13.0% | 13.0% | 13.0% | 0.0% | 2.0% | 0.0% | 0.0% | 72.0% | 11.0% | 45.3% | 25.0% | 34.2 | 37.5 |
| fragile | 53.0% (43.3%–62.5%) | 53.0% | 9.0% | 9.0% | 0.0% | 6.0% | 4.0% | 0.0% | 28.0% | 13.0% | 10.8% | 55.8% | 43.4 | 43.1 |
| wind flood | 5.0% (2.2%–11.2%) | 4.0% | 5.0% | 5.0% | 0.0% | 0.0% | 0.0% | 0.0% | 90.0% | 5.0% | 62.2% | 12.8% | 31.6 | 35.7 |
| low battery deadline | 44.0% (34.7%–53.8%) | 42.0% | 2.0% | 2.0% | 0.0% | 2.0% | 2.0% | 0.0% | 50.0% | 4.0% | 33.9% | 46.8% | 31.3 | 37.0 |
| compound | 3.0% (1.0%–8.5%) | 3.0% | 4.0% | 4.0% | 0.0% | 1.0% | 0.0% | 0.0% | 92.0% | 4.0% | 79.2% | 8.3% | 32.0 | 34.9 |
| all combined | 0.0% (0.0%–3.7%) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 0.0% | 100.0% | 0.0% | – | – |
| low forecast reliability | 69.0% (59.4%–77.2%) | 69.0% | 11.0% | 11.0% | 0.0% | 3.0% | 4.0% | 0.0% | 13.0% | 15.0% | 5.5% | 72.6% | 35.1 | 39.1 |
| **All profiles** | 38.8% (36.0%–41.7%) | 38.5% | 6.1% | 6.1% | 0.0% | 2.1% | 2.1% | 0.0% | 50.9% | 7.9% | 37.5% | 51.2% | 35.0 | 38.9 |

Best profile: **normal** (71.0%); hardest: **all combined** (0.0%).

## 5. Full method vs immediate safety filter, per profile

Paired differences (full − immediate) on identical missions, with 95 % paired-bootstrap CIs. A CI that excludes 0 is a reliable difference.

| Profile | Δ Success | Δ Success+rec. | Δ UAV lost |
|---|---:|---:|---:|
| normal | +2.0 pp [-3.0, +7.0] | +2.0 pp [-3.0, +7.0] | -6.0 pp [-11.0, -2.0] |
| low battery | -3.0 pp [-10.0, +4.0] | +7.0 pp [+2.0, +12.0] | -18.0 pp [-26.0, -10.0] |
| tight deadline | +4.0 pp [+0.0, +9.0] | +3.0 pp [-1.0, +7.0] | -5.0 pp [-10.0, +0.0] |
| strong wind | +3.0 pp [-2.0, +8.0] | +3.0 pp [-2.0, +8.0] | -12.0 pp [-20.0, -4.0] |
| dynamic flood | +3.0 pp [+0.0, +7.0] | +3.0 pp [+0.0, +7.0] | -53.0 pp [-63.0, -43.0] |
| fragile | +4.0 pp [-1.0, +10.0] | +4.0 pp [-1.0, +10.0] | -8.0 pp [-14.0, -3.0] |
| wind flood | -2.0 pp [-6.0, +2.0] | -3.0 pp [-7.0, +0.0] | -62.0 pp [-72.0, -52.0] |
| low battery deadline | -6.0 pp [-14.0, +2.0] | +3.0 pp [-3.0, +10.0] | -9.0 pp [-16.0, -3.0] |
| compound | -1.0 pp [-4.0, +2.0] | +0.0 pp [-3.0, +3.0] | -43.0 pp [-53.0, -33.0] |
| all combined | +0.0 pp [+0.0, +0.0] | +0.0 pp [+0.0, +0.0] | -53.0 pp [-63.0, -43.0] |
| low forecast reliability | +1.0 pp [-2.0, +4.0] | +1.0 pp [-2.0, +4.0] | -6.0 pp [-12.0, +0.0] |
| **All profiles** | +0.5 pp [-1.1, +2.0] | +2.1 pp [+0.8, +3.5] | -25.0 pp [-27.6, -22.4] |

## 6. Discussion

### 6.1 Why the pooled success is about 39 %

* The pooled figure averages **11 profiles of very different difficulty**. On the 7 profiles where at least 90 % of missions are oracle-feasible (normal, low battery, tight deadline, strong wind, fragile, low battery deadline, low forecast reliability), the full method delivers **58.0%** on average. On the 4 disaster profiles (dynamic flood, wind flood, compound, all combined) it delivers 5.2%, where only 43.2% of missions are possible even with perfect foresight.
* **The ceiling is far below 100 %.** Only 75.8% of all test missions are oracle-feasible, and that oracle ignores two requirements the real system must meet: a fragile payload must arrive above 0.80 stability (which can require many re-stabilising hover steps), and the UAV must still reach a landing site after delivery. The achievable ceiling is therefore lower than this.
* **On missions that are possible, the full method delivers 51.2%.** The remaining gap comes from the learned policy: it sees only a 17×17 window, not the whole city, and it was fine-tuned for 2 M steps. The raw PPO+CNN+LSTM policy delivers 33.4% overall.
* **Most remaining aborts are correct decisions.** The full method aborts 50.9% of missions. Inspected aborts on oracle-feasible missions were mainly (a) fragile payloads that could not be re-stabilised before the deadline and (b) low-battery missions where delivery was likely but no landing site was reachable afterwards. Both are 'unrecoverable' by the research definition.

### 6.2 Would a larger 'step size' improve it?

| Kind of step | Effect | Evidence |
|---|---|---|
| More training steps | Small gains only | Training success rose slowly (≈0.32 → 0.36 over 2 M steps); PPO beat its behaviour-cloned start by +4.1 pp. The oracle ceiling caps any gain. |
| Larger learning-rate step | Worse | A too-aggressive update destroyed the cloned policy (≈37 % → 4 % success within 10 updates). Fine-tuning therefore uses lr 1e-4, a critic warm-up and a KL anchor. |
| Larger simulation time step (more weather per decision) | Worse | The flood advances further between decisions, so there are fewer chances to react. A smaller step looks easier but changes the problem, not the method. |
| More deadline steps | Higher success | Tight deadline 64.0% vs normal 71.0%. This also changes the task, not the method. |

### 6.3 What the numbers mean for a real deployment

* **As a research result, the hypothesis is supported.** Against the conventional immediate filter, UAV losses fall from 31.1% to 6.1%, and failures that slip past the one-step check fall from 27.1% to 7.9%, while delivered-and-recoverable missions do not decrease (see section 5).
* **The pooled 38.8% is not a real-world delivery rate.** The flood model is synthetic and uncalibrated, and operational medical-drone services cancel or postpone flights in severe weather instead of flying into it. The closest analogue is the normal profile: 71.0% delivered, with a safe decision in most other cases. Refusing to launch into a hopeless mission is what a real operator would do.
* **The loss rate is the main gap to real-world use.** 7.0% UAV loss in normal missions is far above what aviation would accept, where vehicle losses must be rare events. Real deployment would need much stronger perception and control, calibrated flood data and certification-level validation.
* **Recommended reporting in the paper:** the paired relative improvements (section 5), success on oracle-feasible missions, and the per-profile tables. Do not headline the pooled success as a real-world performance figure.
