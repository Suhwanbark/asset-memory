"""Create a versioned article-boundary correction from existing archived HTML.

No network or model calls. Original source/case/gold files are never rewritten.
This is specifically the audited Amazon article, not claim-based text selection.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/temporal_evidence_poc_v1"


def sha(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def normalized_text(node):
    return node.get_text("\n", strip=True).replace("\xa0", " ").strip() + "\n"


def exclusive_write(path, value):
    with path.open("xb") as stream:
        stream.write(value.encode() if isinstance(value, str) else value)


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-id", default="corp_irbt_20240129")
    parser.add_argument("--new-source-id", default="corp_irbt_20240129_clean_v3")
    parser.add_argument("--removal-selector", default=".articles-links")
    args = parser.parse_args()
    if args.source_id != "corp_irbt_20240129" or args.removal_selector != ".articles-links":
        raise ValueError("This correction is validated only for the audited Amazon source/module")
    if not args.new_source_id.startswith(args.source_id + "_clean_"):
        raise ValueError("A separate versioned clean source ID is required")
    old_folder = DATA / "sources" / args.source_id
    new_folder = DATA / "sources" / args.new_source_id
    patch_path = DATA / "corporate_clean_case_patch.json"
    if new_folder.exists() or patch_path.exists():
        raise FileExistsError("Preserving existing artifacts; choose a new source/patch version explicitly")
    meta = json.loads((old_folder / "meta.json").read_text())
    raw = Path(meta["raw_path"]).read_bytes()
    old_text = Path(meta["text_path"]).read_text()
    assert sha(raw) == meta["raw_sha256"]
    assert sha(old_text) == meta["text_sha256"]

    soup = BeautifulSoup(raw, "html.parser")
    body = soup.select_one(meta["main_selector"])
    assert body is not None
    for node in body.select("script,style,svg,noscript"):
        node.decompose()
    assert normalized_text(body) == old_text
    modules = body.select(args.removal_selector)
    assert len(modules) == 1, "Expected exactly one explicit recommendation module"
    module = modules[0]
    assert module.select_one(".articles-links__title").get_text(strip=True) == "Trending news and stories"
    article = body.select_one(".ArticlePage-articleBody > article")
    children = article.find_all(recursive=False)
    assert len(children) == 5
    assert all("block" in node.get("class", []) for node in children[:4])
    assert children[-1].select_one(args.removal_selector) is module
    original_article_blocks = [normalized_text(node) for node in children[:4]]
    removed_text = normalized_text(module)
    removed_html = str(module)
    removed_links = [{"label": node.get_text(" ", strip=True), "url": node["href"]}
                     for node in module.select("a[href]")]
    assert "2026" in removed_text
    module.decompose()
    clean_text = normalized_text(body)
    assert all(part.strip() in clean_text for part in original_article_blocks)
    assert "".join(c for c in old_text if c.isalnum()) == "".join(
        c for c in clean_text + removed_text if c.isalnum())
    assert "Trending news and stories" not in clean_text and "2026" not in clean_text
    assert 2100 < len(clean_text) < len(old_text)
    for phrase in meta["completeness_anchors"]:
        assert phrase.casefold() in clean_text.casefold()

    created_at = datetime.now(timezone.utc).isoformat()
    new_meta = dict(meta)
    new_meta.update(
        id=args.new_source_id,
        raw_path=str(new_folder / "source.html"),
        text_path=str(new_folder / "source.txt"),
        text_sha256=sha(clean_text), characters=len(clean_text),
        source_alphanumeric_sha256=sha("".join(c for c in clean_text if c.isalnum())),
        all_main_document_alphanumeric_characters_preserved=False,
        all_article_body_alphanumeric_characters_preserved=True,
        scope="Complete four-block publisher article; explicit sibling trending-news module excluded by DOM role, not by claim content",
        excluded_module_selectors=[args.removal_selector],
        derived_from_source_id=args.source_id,
        derived_from_text_sha256=meta["text_sha256"],
        derivation_at=created_at,
        source_snapshot_unchanged_since_publication_verified=False,
    )
    patch = dict(
        version="clean_v3", old_case_id="corporate_amazon_irobot",
        old_id=args.source_id, new_doc_metadata=new_meta,
        removal_selector=args.removal_selector,
        removed_text=removed_text, removed_exacttext=removed_text,
        removed_text_sha256=sha(removed_text), removed_html_sha256=sha(removed_html),
        removed_links=removed_links,
        before_characters=len(old_text), after_characters=len(clean_text),
        removed_characters=len(removed_text),
        source_evidence=dict(
            original_meta_path=str(old_folder / "meta.json"),
            original_raw_path=meta["raw_path"], original_raw_sha256=sha(raw),
            original_text_path=meta["text_path"], original_text_sha256=sha(old_text),
            module_heading="Trending news and stories",
            structural_boundary=".ArticlePage-articleBody > article: four div.contentContainer.block article children, then one div.contentContainer recommendation child",
            original_article_block_count=4,
            original_article_blocks_sha256=[sha(part) for part in original_article_blocks],
            unchanged_article_blocks_verified=True,
            raw_bytes_copied_without_modification=True,
            network_refetch=False,
        ),
        generated_at=created_at, generator_sha256=sha(Path(__file__).read_bytes()),
        correction_basis="Post-run source audit; source-boundary correction independent of model answers. Original gold claims and labels unchanged.",
        original_cases_and_gold_modified=False,
        require_rebuild_and_reevaluate_affected_case=True,
        excluded_separate_case=dict(id="corporate_vaneck_spot_bitcoin", reason="January-11-dated PDF artifact created/modified April22–23, after January12 cutoff; no contemporaneous artifact verification"),
    )
    # Exclusive writes prevent silently replacing any previous source or result.
    new_folder.mkdir(parents=True, exist_ok=False)
    exclusive_write(new_folder / "source.html", raw)
    exclusive_write(new_folder / "source.txt", clean_text)
    exclusive_write(new_folder / "meta.json", json_bytes(new_meta))
    exclusive_write(patch_path, json_bytes(patch))
    assert Path(meta["raw_path"]).read_bytes() == raw
    assert Path(meta["text_path"]).read_text() == old_text
    print(json.dumps({"patch_path": str(patch_path), "source_id": args.new_source_id,
                      "before_characters": len(old_text), "after_characters": len(clean_text),
                      "removed_characters": len(removed_text), "text_sha256": sha(clean_text)}))


if __name__ == "__main__":
    main()
