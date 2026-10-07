from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from .runtime import ROOT, check_disk, cpu_only, watchdog, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["preflight", "freeze-cohort", "run-dev", "grade-dev",
                                          "freeze-method", "run-evaluation", "grade-evaluation"])
    parser.add_argument("--run-id", default="finrate_luna_v1")
    parser.add_argument("--policy", default=str(ROOT / "data/finrate_luna_protocol_v1.json"))
    parser.add_argument("--method-version", default="v1")
    args = parser.parse_args()
    if not args.run_id.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Invalid run ID")
    if not args.method_version.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Invalid method version")
    cpu_only()
    check_disk()
    out = ROOT / "result" / args.run_id
    status = out / "status.json"
    stop = watchdog(status)
    write_json(status, {"state": "running_" + args.action, "pid": os.getpid(),
                        "started": time.time(), "gpu_allowed": False,
                        "stop_below_free_bytes": 10_000_000_000})
    try:
        from .data import acquire, audit
        folder = ROOT / "data/finrate_public_v1"
        if args.action == "preflight":
            acquire(folder)
            report = audit(folder, out)
            write_json(status, {**report, "pid": os.getpid(), "finished": time.time(),
                                "inference_calls": 0, "gpu_allowed": False,
                                "free_disk": check_disk()})
            print(json.dumps(report, indent=2), flush=True)
            # No repeated paid polling and no silently weakened cutoff condition.
            return 0 if report["date_candidates"] else 20
        from .cohort import approve_gate, freeze
        report = json.loads((out / "cutoff_audit.json").read_text())
        policy = json.loads(Path(args.policy).read_text())
        approve_gate(report, policy)
        cohort = out / "cohort"
        if args.action == "freeze-cohort":
            source_metadata = json.loads((out / "document_metadata.json").read_text())
            approved_path = out / "dependency_approvals.json"
            approvals = json.loads(approved_path.read_text()) if approved_path.exists() else {}
            freeze(folder, cohort, report, policy, source_metadata=source_metadata,
                   dependency_approvals=approvals)
        elif args.action in {"run-dev", "run-evaluation"}:
            from .experiment import run
            run(policy_path=args.policy, audit_path=out / "cutoff_audit.json", cohort_dir=cohort,
                output_dir=out, split=args.action.removeprefix("run-"), method_version=args.method_version)
        elif args.action == "freeze-method":
            from .experiment import freeze_method
            freeze_method(args.policy, cohort, out, args.method_version)
        else:
            from .scoring import grade
            grade(cohort, out, split=args.action.removeprefix("grade-"), version=args.method_version)
        write_json(status, {"state": "completed_" + args.action, "pid": os.getpid(),
                            "finished": time.time(), "gpu_allowed": False, "free_disk": check_disk()})
        return 0
    except Exception as exc:
        write_json(status, {"state": "failed_" + args.action, "reason": str(exc),
                            "pid": os.getpid(), "finished": time.time()})
        raise
    finally:
        stop.set()


if __name__ == "__main__":
    raise SystemExit(main())
