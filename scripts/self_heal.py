"""Entrypoint: run one self-healing monitoring cycle.

This is the demo that shows the whole system working end to end.

Examples
--------
# Simulate a medium drift and see what the agent decides (advisory only):
    python -m scripts.self_heal --simulate medium

# Let the system actually heal itself (retrain/rollback for real):
    python -m scripts.self_heal --simulate high --auto

# Use the REAL production window from the serving prediction log instead:
    python -m scripts.self_heal --from-log
"""

from __future__ import annotations

import argparse
import logging

from mlops.data.generate import make_drift_spec
from mlops.orchestration.self_healing import SelfHealingMonitor


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one self-healing cycle.")
    parser.add_argument(
        "--simulate",
        choices=["none", "low", "medium", "high"],
        default="medium",
        help="inject this drift severity into a generated production window",
    )
    parser.add_argument(
        "--from-log",
        action="store_true",
        help="use the real serving prediction log as the window (if present)",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="actually execute remediation (retrain/rollback); default is advisory",
    )
    parser.add_argument("--seed", type=int, default=7, help="seed for the simulated window")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    monitor = SelfHealingMonitor(auto_remediate=args.auto)

    # Choose the production window.
    window = None
    if args.from_log:
        window = monitor.window_from_log()
        if window is None:
            print("No prediction log found — falling back to simulation.")
    if window is None:
        window = monitor.window_from_simulation(
            make_drift_spec(args.simulate), seed=args.seed
        )

    print("=" * 70)
    print(f"Self-healing cycle  (mode={'auto' if args.auto else 'advisory'}, "
          f"window={len(window)} rows)")
    print("=" * 70)

    result = monitor.run_cycle(window)

    print("\n" + result.summary())
    print("\n(advisory mode: re-run with --auto to actually remediate)"
          if not args.auto else "\n(auto mode: remediation executed above)")


if __name__ == "__main__":
    main()
