"""Entrypoint: train a model and register/promote it.

Usage:
    python -m scripts.run_pipeline
    python -m scripts.run_pipeline --rows 10000

This is the "initial training" you run once to bootstrap the system, and the
same code path the self-healing loop re-invokes to RETRAIN.
"""

from __future__ import annotations

import argparse

from mlops.training.train import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train + register the fraud model.")
    parser.add_argument("--rows", type=int, default=8000, help="training dataset size")
    parser.add_argument("--seed", type=int, default=None, help="override data seed")
    args = parser.parse_args()

    print("=" * 70)
    print("Training self-healing MLOps model")
    print("=" * 70)

    outcome = train_model(n_rows=args.rows, data_seed=args.seed)

    ev = outcome.evaluation
    print(f"\nRun ID:        {outcome.run_id}")
    print("Metrics:")
    print(f"  accuracy   {ev.accuracy:.3f}")
    print(f"  precision  {ev.precision:.3f}")
    print(f"  recall     {ev.recall:.3f}")
    print(f"  f1         {ev.f1:.3f}")
    print(f"  roc_auc    {ev.roc_auc:.3f}")
    print(f"\nQuality gate:  {ev.gate_reason}")
    print(f"Registered:    version {outcome.registered_version}")
    print(f"Promotion:     {outcome.promotion_note}")
    print("\nDone. Serve it with:  uvicorn mlops.serving.app:app --reload")


if __name__ == "__main__":
    main()
