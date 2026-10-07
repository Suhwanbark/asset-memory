"""Run only frozen, explicitly eligible cohorts; never read gold at inference."""
from __future__ import annotations

import json
from pathlib import Path

from .agent import predict
from .bank import Bank
from .cohort import approve_gate
from .data import documents, metadata
from .memory import build_asset_memory
from .model import Luna
from .retrieval import Index
from .runtime import ROOT, check_disk, digest, file_digest, write_json


def run(*, policy_path, audit_path, cohort_dir, output_dir, split="dev", method_version="v1"):
    if split not in {"dev", "evaluation"}:
        raise ValueError(split)
    policy = json.loads(Path(policy_path).read_text())
    audit = json.loads(Path(audit_path).read_text())
    approve_gate(audit, policy)
    cohort = Path(cohort_dir)
    manifest = json.loads((cohort / "cohort_manifest.json").read_text())
    if manifest["policy"] != policy:
        raise ValueError("Policy changed after cohort freezing")
    inputs = json.loads((cohort / f"{split}_inputs.json").read_text())
    if not inputs:
        raise ValueError("Empty evaluation is not a successful experiment")
    if any(set(q) != {"q_id", "question", "task"} for q in inputs):
        raise ValueError("Unexpected inference fields: possible answer/gold leakage")
    out = Path(output_dir)
    code_hashes = {p.name: file_digest(p) for p in Path(__file__).parent.glob("*.py")}
    frozen = {"method_version": method_version, "code_hashes": code_hashes, "policy": policy,
              "cohort_sha256": file_digest(cohort / "cohort_manifest.json")}
    if split == "evaluation":
        release = out / method_version / "method_frozen_for_evaluation.json"
        if not release.exists() or json.loads(release.read_text()) != frozen:
            raise ValueError("Explicitly freeze the method before accessing evaluation inputs")
        write_json(out / "evaluation_lock.json", {
            "cohort_sha256": frozen["cohort_sha256"], "method_version": method_version,
            "code_hashes": code_hashes}, immutable=True)
    write_json(out / method_version / split / "run_manifest.json", frozen, immutable=True)
    model = Luna(reasoning=policy.get("model_reasoning_effort", "low"))
    docs = [d for d in documents(ROOT / "data/finrate_public_v1") if metadata(d)["ticker"] in manifest["assets"]]
    print("INDEX_BUILD", len(docs), flush=True)
    index = Index.build(docs, out / "index")
    print("INDEX_READY", len(index.passages), flush=True)

    def run_conditions(conditions, banks):
        for q in inputs:
            for condition in conditions:
                check_disk()
                folder = out / method_version / split / q["q_id"] / condition
                result_path = folder / "prediction.json"
                if result_path.exists():
                    continue
                try:
                    result = predict(q["question"], condition, index, docs, banks, model, folder,
                                     evidence_budget=policy["evidence_tokens"],
                                     token_ceiling=policy["online_token_ceiling"], steps=policy["agent_max_steps"])
                    record = {"q_id": q["q_id"], "condition": condition, "status": "completed", **result}
                except Exception as exc:
                    # Failed outputs stay in the denominator. Never retry to get a better answer.
                    record = {"q_id": q["q_id"], "condition": condition, "status": "failed",
                              "error": type(exc).__name__ + ": " + str(exc)}
                write_json(result_path, record, immutable=True)
                print("PREDICTION", split, q["q_id"], condition, record["status"], flush=True)

    run_conditions([c for c in policy["conditions"] if c != "agentic"], {})
    if "agentic" not in policy["conditions"]:
        return
    build_asset_memory(docs, out / "asset_memory", model, version=method_version)
    banks = {asset: Bank(out / "asset_memory" / method_version / asset / "bank.sqlite", readonly=True)
             for asset in manifest["assets"]}
    try:
        run_conditions(["agentic"], banks)
    finally:
        for bank in banks.values():
            bank.close()


def freeze_method(policy_path, cohort_dir, output_dir, method_version):
    policy = json.loads(Path(policy_path).read_text())
    cohort = Path(cohort_dir)
    out = Path(output_dir)
    inputs = json.loads((cohort / "dev_inputs.json").read_text())
    for q in inputs:
        for condition in policy["conditions"]:
            if not (out / method_version / "dev" / q["q_id"] / condition / "prediction.json").exists():
                raise ValueError("Development grid is incomplete")
    if not (out / method_version / "dev" / "report.json").exists():
        raise ValueError("Development assessment is required before freezing")
    write_json(out / method_version / "method_frozen_for_evaluation.json", {
        "method_version": method_version,
        "code_hashes": {p.name: file_digest(p) for p in Path(__file__).parent.glob("*.py")},
        "policy": policy, "cohort_sha256": file_digest(cohort / "cohort_manifest.json")}, immutable=True)
