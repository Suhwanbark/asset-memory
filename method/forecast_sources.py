#!/usr/bin/env python3
"""Acquire fixed primary-source releases, preserving their complete main bodies.

This module performs no model calls and never reads the separate gold file.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "forecast_memory_smoke_v1"
CASES = [
    {
        "id": "nvidia_2022_q2",
        "company": "NVIDIA",
        "past_date": "2022-05-25",
        "current_date": "2022-08-24",
        "past_url": "https://nvidianews.nvidia.com/news/nvidia-announces-financial-results-for-first-quarter-fiscal-2023",
        "current_url": "https://nvidianews.nvidia.com/news/nvidia-announces-financial-results-for-second-quarter-fiscal-2023",
        "past_selector": ".article-body",
        "current_selector": ".article-body",
    },
    {
        "id": "intel_2024_q2",
        "company": "Intel",
        "past_date": "2024-04-25",
        "current_date": "2024-08-01",
        "past_url": "https://www.intc.com/filings-reports/all-sec-filings/content/0000050863-24-000074/q124_earningsrelease.htm",
        "current_url": "https://www.intc.com/news-events/press-releases/detail/1704/intel-reports-second-quarter-2024-financial-results",
        "past_selector": "body",
        "current_selector": "article.full-news-article",
    },
    {
        "id": "amd_2022_q3",
        "company": "AMD",
        "past_date": "2022-08-02",
        "current_date": "2022-11-01",
        "past_url": "https://ir.amd.com/news-events/press-releases/detail/1084/amd-reports-second-quarter-2022-financial-results",
        "current_url": "https://ir.amd.com/news-events/press-releases/detail/1098/amd-reports-third-quarter-2022-financial-results",
        "past_selector": "article.full-news-article",
        "current_selector": "article.full-news-article",
    },
]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def extract_release(html: bytes, selector: str) -> tuple[str, int]:
    soup = BeautifulSoup(html, "html.parser")
    body = soup.select_one(selector)
    if body is None:
        raise ValueError(f"Missing complete-release selector: {selector}")
    for tag in body.select("script, style, noscript, iframe, svg"):
        tag.decompose()
    tables = [t for t in body.find_all("table") if t.find_parent("table") is None]
    table_count = len(tables)
    for table in tables:
        rows = []
        for tr in table.find_all("tr"):
            if tr.find_parent("table") is not table:
                continue
            values = []
            for cell in tr.find_all(["th", "td"], recursive=False):
                value = re.sub(r"\s+", " ", cell.get_text(" ", strip=True)).strip()
                span = int(cell.get("colspan", 1))
                values.extend([value.replace("|", "\\|")] + [""] * (span - 1))
            if values:
                rows.append(" | ".join(values))
        table.replace_with(soup.new_string("\n[TABLE]\n" + "\n".join(rows) + "\n[/TABLE]\n"))
    text = body.get_text("\n", strip=True).replace("\xa0", " ")
    text = re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"
    if len(text) < 10000:
        raise ValueError(f"Unexpectedly short release ({len(text)} chars); inspect possible CDN response")
    return text, table_count


def acquire(case: dict, phase: str) -> tuple[str, str, dict]:
    url = case[f"{phase}_url"]
    response = requests.get(url, timeout=60, headers={"User-Agent": "ResearchSourceArchive/1.0"})
    response.raise_for_status()
    if "html" not in response.headers.get("Content-Type", ""):
        raise ValueError(f"Unexpected content type at {url}: {response.headers.get('Content-Type')}")
    source_bytes = response.content
    text, table_count = extract_release(source_bytes, case[f"{phase}_selector"])
    path = OUT / case["id"]
    path.mkdir(parents=True, exist_ok=True)
    html_path = path / f"{phase}.html"
    text_path = path / f"{phase}.txt"
    html_path.write_bytes(source_bytes)
    text_path.write_text(text, encoding="utf-8")
    return case["id"], phase, {
        "url": url,
        "resolved_url": response.url,
        "publication_date": case[f"{phase}_date"],
        "publisher": case["company"],
        "source_type": "official_company_earnings_release",
        "html_path": str(html_path),
        "text_path": str(text_path),
        "html_sha256": digest(source_bytes),
        "text_sha256": digest(text.encode("utf-8")),
        "html_bytes": len(source_bytes),
        "text_characters": len(text),
        "table_count": table_count,
        "http_status": response.status_code,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "parse_rule": {
            "selector": case[f"{phase}_selector"],
            "scope": "Complete main release including financial tables, outlook, reconciliations and cautionary language; no claim-based shortening",
            "remove": ["script", "style", "noscript", "iframe", "svg"],
            "tables": "All top-level tables converted to row-wise pipe-delimited text with colspan placeholders; nested table text retained in cells",
        },
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gathered = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(acquire, case, phase) for case in CASES for phase in ("past", "current")]
        for future in concurrent.futures.as_completed(futures):
            case_id, phase, source = future.result()
            gathered[(case_id, phase)] = source
            print(f"Fetched {case_id}/{phase}: {source['text_characters']} chars, {source['table_count']} tables", flush=True)
    manifest = {
        "dataset_version": "forecast_memory_smoke_v1",
        "selection": "Three fixed real release pairs chosen before model runs; no selection based on model or memory performance",
        "model_input_boundary": "Only past release belongs to memory formation; current release belongs to follow-up evaluation. gold.json is evaluator-only and must not be read by any agent or memory-formation input builder.",
        "cases": [],
    }
    for case in CASES:
        item = {k: case[k] for k in ("id", "company", "past_date", "current_date")}
        item.update({phase: gathered[(case["id"], phase)] for phase in ("past", "current")})
        manifest["cases"].append(item)
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
