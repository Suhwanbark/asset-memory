"""One resumable development cycle, including grading; never auto-peek at holdout."""
from __future__ import annotations

import argparse
import os
import time

from .runtime import ROOT, check_disk, cpu_only, watchdog, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v1")
    args = parser.parse_args()
    if not args.version.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Invalid version")
    cpu_only()
    check_disk()
    os.nice(10)
    out = ROOT / "result/finrate_luna_v1"
    status = out / "status.json"
    stop = watchdog(status)
    base = {"pid": os.getpid(), "started": time.time(), "method_version": args.version,
            "gpu_allowed": False, "stop_below_free_bytes": 10_000_000_000,
            "cutoff_policy": "exploratory_exposure_uncontrolled"}
    write_json(status, {**base, "state": "running_development"})
    try:
        from .experiment import run
        from .scoring import grade
        run(policy_path=ROOT / "data/finrate_luna_protocol_v1.json", audit_path=out / "cutoff_audit.json",
            cohort_dir=out / "cohort", output_dir=out, split="dev", method_version=args.version)
        write_json(status, {**base, "state": "grading_development"})
        report = grade(out / "cohort", out, split="dev", version=args.version)
        # Development inspection, ablations and method revision follow. No
        # assertion of success based solely on an agentic > baseline score.
        write_json(status, {**base, "state": "development_ready_for_review",
                            "finished": time.time(), "summary": report["summary"],
                            "evaluation_touched": False, "free_disk": check_disk()})
        print("DEVELOPMENT_READY_FOR_REVIEW", flush=True)
    except Exception as exc:
        write_json(status, {**base, "state": "development_failed", "finished": time.time(),
                            "error": type(exc).__name__ + ": " + str(exc)})
        raise
    finally:
        stop.set()


if __name__ == "__main__":
    main()
