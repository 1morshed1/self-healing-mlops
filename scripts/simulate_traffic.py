"""Entrypoint: send synthetic traffic to a running serving instance.

This populates the real prediction log (`logs/predictions.jsonl`) so the
`--from-log` path of the self-healing monitor has genuine production data to
analyze — closing the loop with the actual API rather than simulated windows.

Start the server first:
    uvicorn mlops.serving.app:app

Then, in another shell:
    python -m scripts.simulate_traffic --n 500 --drift medium
"""

from __future__ import annotations

import argparse

import requests

from mlops.config import get_settings
from mlops.data.generate import generate_dataset, make_drift_spec
from mlops.data.schema import FEATURE_NAMES


def main() -> None:
    parser = argparse.ArgumentParser(description="Send synthetic transactions to /predict.")
    parser.add_argument("--n", type=int, default=500, help="number of transactions to send")
    parser.add_argument(
        "--drift",
        choices=["none", "low", "medium", "high"],
        default="none",
        help="drift severity to bake into the generated traffic",
    )
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    settings = get_settings()
    base = f"http://{settings.serve_host}:{settings.serve_port}"
    if settings.serve_host == "0.0.0.0":
        base = f"http://127.0.0.1:{settings.serve_port}"

    data = generate_dataset(args.n, seed=args.seed, drift=make_drift_spec(args.drift))
    sent, fraud = 0, 0
    for _, row in data.iterrows():
        features = {name: float(row[name]) for name in FEATURE_NAMES}
        try:
            resp = requests.post(f"{base}/predict", json={"features": features}, timeout=10)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise SystemExit(
                f"Request failed — is the server running at {base}? ({exc})"
            ) from exc
        sent += 1
        fraud += resp.json()["is_fraud"]

    print(f"Sent {sent} transactions ({args.drift} drift). "
          f"Model flagged {fraud} as fraud ({fraud / sent:.1%}).")
    print(f"Predictions logged to {settings.prediction_log}. "
          f"Now run:  python -m scripts.self_heal --from-log")


if __name__ == "__main__":
    main()
