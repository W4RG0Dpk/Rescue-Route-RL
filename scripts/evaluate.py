"""
Benchmark evaluation.

    python scripts/evaluate.py --stage tune      # pick delta on VALIDATION seeds
    python scripts/evaluate.py --stage test      # final tables on held-out TEST seeds
    python scripts/evaluate.py --stage all

Outputs in reports/evaluation/: episodes.csv, summary.csv, results.json,
evaluation_report.md and figures/.
"""

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ALL_PROFILES = ["normal", "low_battery", "tight_deadline", "strong_wind", "dynamic_flood", "fragile",
                "wind_flood", "low_battery_deadline", "compound", "all_combined", "low_forecast_reliability"]
TUNE_PROFILES = ["normal", "dynamic_flood", "low_battery", "compound"]
ALL_METHODS = ["random", "bc_cnn_lstm", "ppo_mlp", "ppo_cnn", "ppo_cnn_lstm",
               "ppo_cnn_lstm_risk", "ppo_cnn_lstm+immediate", "ppo_cnn_lstm+deterministic", "ppo_cnn_lstm+probabilistic"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["tune", "test", "all"])
    ap.add_argument("--episodes", type=int, default=100, help="test seeds per profile")
    ap.add_argument("--tune-episodes", type=int, default=16, help="validation seeds per profile")
    ap.add_argument("--processes", type=int, default=28)
    ap.add_argument("--deltas", default="0.2,0.3,0.4")
    ap.add_argument("--abort-thresholds", default="0.5,0.7,0.85")
    ap.add_argument("--loss-budget", type=float, default=0.08,
                    help="maximum UAV-loss rate allowed when choosing the shield thresholds")
    ap.add_argument("--methods", default=",".join(ALL_METHODS))
    ap.add_argument("--profiles", default=",".join(ALL_PROFILES))
    args = ap.parse_args()

    import pandas as pd

    from rescue_route.evaluation.experiments import TEST_SEED0, VAL_SEED0, run_grid, summarize

    out = ROOT / "reports" / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    delta_file = out / "delta.json"

    if args.stage in ("tune", "all"):
        rows = []
        for d in [float(x) for x in args.deltas.split(",")]:
            for da in [float(x) for x in args.abort_thresholds.split(",")]:
                print(f"== tuning delta={d}, delta_abort={da} on validation seeds", flush=True)
                res = run_grid(["ppo_cnn_lstm+probabilistic"], ALL_PROFILES,
                               range(VAL_SEED0, VAL_SEED0 + args.tune_episodes), delta=d, processes=args.processes,
                               delta_abort=da)
                df = pd.DataFrame(res)
                rows.append({"delta": d, "delta_abort": da, "success": float(df["success"].mean()),
                             "success_and_recovery": float(df["success_and_recovery"].mean()),
                             "uav_lost": float(df["uav_lost"].mean()), "safe_abort": float(df["safe_abort"].mean()),
                             "objective": float(df["success_and_recovery"].mean())})
                print(rows[-1], flush=True)
        tune = pd.DataFrame(rows)
        tune.to_csv(out / "delta_tuning.csv", index=False)
        ok = tune[tune["uav_lost"] <= args.loss_budget]
        pick = (ok if len(ok) else tune.sort_values("uav_lost").head(1))
        best = pick.sort_values(["objective", "uav_lost"], ascending=[False, True]).iloc[0]
        delta_file.write_text(json.dumps({
            "delta": float(best["delta"]), "delta_abort": float(best["delta_abort"]),
            "objective": f"maximise mean(success_and_recovery) subject to UAV-loss rate <= {args.loss_budget}",
            "validation_seeds": [VAL_SEED0, VAL_SEED0 + args.tune_episodes - 1],
            "profiles": ALL_PROFILES, "table": rows}, indent=2))
        print(f"selected delta = {best['delta']}, delta_abort = {best['delta_abort']}")

    if args.stage in ("test", "all"):
        tuned = json.loads(delta_file.read_text()) if delta_file.exists() else {}
        delta, delta_abort = tuned.get("delta", 0.15), tuned.get("delta_abort", 0.5)
        methods = args.methods.split(",")
        profiles = args.profiles.split(",")
        print(f"== test: {len(methods)} methods x {len(profiles)} profiles x {args.episodes} seeds (delta={delta})")
        res = run_grid(methods, profiles, range(TEST_SEED0, TEST_SEED0 + args.episodes), delta=delta,
                       processes=args.processes, delta_abort=delta_abort)
        df = pd.DataFrame(res)
        df.to_csv(out / "episodes.csv", index=False)
        summarize(df).to_csv(out / "summary.csv", index=False)
        summarize(df, by=("method",)).to_csv(out / "summary_overall.csv", index=False)

        from rescue_route.evaluation.report import write_evaluation_report

        write_evaluation_report(out, df, delta)

        from rescue_route.evaluation.profile_report import write_profile_report

        print("Per-profile report:", write_profile_report(out, df))
        print("Report:", out / "evaluation_report.md")


if __name__ == "__main__":
    main()
