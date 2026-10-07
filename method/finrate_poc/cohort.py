"""Outcome-blind cohort freezing and gold-free inference inputs."""
from __future__ import annotations

from collections import defaultdict
from datetime import date
import json
from pathlib import Path

from .data import MODEL, MODEL_CUTOFF, TASKS
from .runtime import digest, write_json


def approve_gate(audit, policy):
    if policy.get("model") != MODEL:
        raise ValueError("Changing the requested model requires an explicit revised protocol")
    mode = policy.get("cutoff_policy", "strict_post_cutoff")
    if mode == "strict_post_cutoff":
        if audit.get("knowledge_cutoff") != MODEL_CUTOFF:
            raise ValueError("Cutoff differs from the verified requested model")
        if not audit.get("date_candidates"):
            raise ValueError("No post-cutoff candidates: inference is prohibited")
        return
    if mode == "exploratory_exposure_uncontrolled" and policy.get("user_authorization"):
        return
    raise ValueError("Changing cutoff policy requires an explicit user decision")


def freeze(folder, out, audit, policy, *, source_metadata, dependency_approvals=None):
    """Fix assets and questions without seeing any model answers or scores.

    Connected components of gold sources stay in one split. Gold is used here
    only by the dataset custodian, never by inference/index/memory code.
    """
    approve_gate(audit, policy)
    lookup = {m["doc_id"]: m for m in source_metadata}
    approved = dependency_approvals or {}
    rows = []
    for task in policy.get("tasks", ["LT-QA"]):
        if task not in TASKS:
            raise ValueError(task)
        for q in json.loads((Path(folder) / f"qa/{task}.json").read_text()):
            ids = q.get("doc_ids") or [q.get("doc_id")]
            if any(i not in lookup or not lookup[i]["filing_date"] for i in ids):
                continue
            if policy.get("cutoff_policy", "strict_post_cutoff") == "strict_post_cutoff":
                if max(lookup[i]["filing_date"] for i in ids) <= MODEL_CUTOFF:
                    continue
                a = approved.get(q["q_id"], {})
                if not a.get("requires_new_post_cutoff_fact") or not a.get("evidence_quote") or not a.get("first_disclosed_at"):
                    continue
                if date.fromisoformat(a["first_disclosed_at"]) <= date.fromisoformat(MODEL_CUTOFF):
                    continue
            assets = tuple(sorted({lookup[i]["ticker"] for i in ids}))
            rows.append({**q, "task": task, "assets": assets, "source_ids": ids})
    if not rows:
        raise ValueError("No approved eligible questions; do not relax the filter automatically")
    total_eligible = len(rows)
    seed = str(policy.get("seed", "finrate-toy-20261004"))
    # Fixed hash order, independent of question content, difficulty and outcomes.
    assets = sorted({a for r in rows for a in r["assets"]}, key=lambda a: digest(seed + a))
    assets = assets[:policy.get("asset_count", 3)]
    rows = [r for r in rows if set(r["assets"]) <= set(assets)]
    parent = {r["q_id"]: r["q_id"] for r in rows}

    def find(q):
        while parent[q] != q:
            parent[q] = parent[parent[q]]
            q = parent[q]
        return q

    owners = {}
    for r in rows:
        for s in r["source_ids"]:
            if s in owners:
                parent[find(r["q_id"])] = find(owners[s])
            owners[s] = r["q_id"]
    groups = defaultdict(list)
    for r in rows:
        groups[find(r["q_id"])].append(r)
    groups = sorted(groups.values(), key=lambda g: digest(seed + min(r["q_id"] for r in g)))
    if len(groups) < 2:
        raise ValueError("Too few independent source groups for development/evaluation")
    split_groups = {"dev": groups[:-max(1, len(groups)//3)],
                    "evaluation": groups[-max(1, len(groups)//3):]}
    result = {}
    for split, gs in split_groups.items():
        candidates = [r for g in gs for r in g]
        candidates.sort(key=lambda r: digest(seed + r["q_id"]))
        result[split] = candidates[:policy.get("dev_count", 20) if split == "dev" else policy.get("eval_count", 10)]
    if not all(result.values()):
        raise ValueError("Empty split")
    doc_sets = [{i for r in result[s] for i in r["source_ids"]} for s in ("dev", "evaluation")]
    assert not doc_sets[0] & doc_sets[1]
    # Frozen input files deliberately omit answers, key_points and gold doc IDs.
    for split, qs in result.items():
        write_json(Path(out) / f"{split}_inputs.json", [
            {"q_id": r["q_id"], "task": r["task"], "question": r["question"]} for r in qs], immutable=True)
        write_json(Path(out) / "gold" / f"{split}.json", qs, immutable=True)
    manifest = {"policy": policy, "assets": assets,
                "selected_ids": {s: [r["q_id"] for r in qs] for s, qs in result.items()},
                "eligible_before_asset_sample": total_eligible,
                "eligible_after_asset_sample": len(parent), "source_group_count": len(groups),
                "no_model_scores_used": True,
                "corpus_rule": "all released documents for selected assets, not only gold evidence"}
    write_json(Path(out) / "cohort_manifest.json", manifest, immutable=True)
    return manifest
