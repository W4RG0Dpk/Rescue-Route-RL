"""
Live RescueRoute Mission Control.

    python scripts/run_demo.py            # http://127.0.0.1:8765

Serves the demo and a small local API so you can simulate ANY method on ANY
profile and seed from the browser (the full recoverability shield takes a few
seconds per mission because it imagines 16 futures at every decision).
Without the server, open src/rescue_route/demo/web/index.html directly for
the recorded showcase missions.
"""

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import webbrowser  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
WEB = ROOT / "src" / "rescue_route" / "demo" / "web"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    from flask import Flask, jsonify, request, send_from_directory

    from rescue_route.env.mission import SCENARIO_PROFILES
    from rescue_route.env.rescue_route_env import RescueRouteEnv
    from rescue_route.evaluation.controllers import METHODS, MODEL_DIR, make_controller
    from rescue_route.evaluation.runner import run_episode

    delta_file = ROOT / "reports" / "evaluation" / "delta.json"
    tuned = json.loads(delta_file.read_text()) if delta_file.exists() else {}
    delta = tuned.get("delta", 0.15)
    delta_abort = tuned.get("delta_abort", 0.5)
    available = [m for m in METHODS if not m.startswith("ppo") or (MODEL_DIR / m.split("+")[0]).exists()]
    lock = threading.Lock()
    app = Flask(__name__, static_folder=None)

    @app.get("/")
    def index():
        return send_from_directory(WEB, "index.html")

    @app.get("/<path:path>")
    def static_files(path):
        return send_from_directory(WEB, path)

    # Known outcomes from the held-out test evaluation (same seeds -> identical, deterministic replays).
    outcomes = {}
    ep_file = ROOT / "reports" / "evaluation" / "episodes.csv"
    if ep_file.exists():
        import pandas as pd

        ep = pd.read_csv(ep_file, usecols=["method", "profile", "seed", "success", "reason"])
        for (m, p), g in ep.groupby(["method", "profile"]):
            outcomes[(m, p)] = {"rate": float(g["success"].mean()),
                                "success_seeds": sorted(int(s) for s in g.loc[g["success"], "seed"])}

    @app.get("/api/meta")
    def meta():
        rates = {f"{m}|{p}": v["rate"] for (m, p), v in outcomes.items()}
        return jsonify({"methods": available, "profiles": list(SCENARIO_PROFILES), "delta": delta, "success_rates": rates})

    @app.get("/api/suggest")
    def suggest():
        """A seed where this method delivered in the test evaluation (falls back to the 'normal' profile)."""
        import random

        method = request.args.get("method", "ppo_cnn_lstm+probabilistic")
        profile = request.args.get("profile", "normal")
        for p in (profile, "normal", "low_forecast_reliability", "tight_deadline"):
            seeds = outcomes.get((method, p), {}).get("success_seeds", [])
            if seeds:
                return jsonify({"profile": p, "seed": random.choice(seeds), "fallback": p != profile,
                                "rate": outcomes[(method, p)]["rate"]})
        return jsonify({"error": "no recorded successful mission for this method"}), 404

    @app.post("/api/run")
    def run():
        body = request.get_json(force=True)
        method = body.get("method", "ppo_cnn_lstm+probabilistic")
        profile = body.get("profile", "normal")
        seed = int(body.get("seed", 2_000_001))
        if method not in available or profile not in SCENARIO_PROFILES:
            return jsonify({"error": "unknown method or profile"}), 400
        with lock:
            env = RescueRouteEnv(profile=profile)
            res = run_episode(env, make_controller(method, seed=seed, delta=delta, delta_abort=delta_abort), seed, profile, record=True, method=method)
        tr = res["trace"]
        tr["delta"] = delta
        return jsonify(tr)

    url = f"http://127.0.0.1:{args.port}"
    print(f"RescueRoute Mission Control: {url}")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
