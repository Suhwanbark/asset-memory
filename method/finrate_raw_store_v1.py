"""Local, immutable filing store; no QA/gold access or model/network calls.

Original HTML bytes are retained. The readable view is a deterministic text
projection, not a claim of visual/PDF equivalence. Tables retain row/cell
separators; original rowspan/colspan and exact DOM ranges remain addressable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from pathlib import Path
from urllib.parse import unquote

from bs4 import BeautifulSoup, NavigableString, Tag

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STORE = ROOT / "data/finrate_raw_smoke10_v1"
BLOCKS = {"div", "p", "li", "ul", "ol", "section", "article", "header", "footer",
          "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre", "table"}
ITEM = re.compile(r"^Item\s+\d+(?:\.\d+)?[A-Z]?\b", re.I)


def original_paths(soup):
    """Addresses in the ORIGINAL BeautifulSoup DOM, before any projection pruning."""
    result = {}

    def visit(node, path):
        result[id(node)] = path
        if not isinstance(node, Tag):
            return
        counts, text_index = {}, 0
        for child in node.children:
            if isinstance(child, Tag):
                counts[child.name] = counts.get(child.name, 0) + 1
                visit(child, f"{path}/{child.name}[{counts[child.name]}]")
            elif type(child) is NavigableString:
                text_index += 1
                visit(child, f"{path}/text()[{text_index}]")

    visit(soup, "")
    return result


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def project_html(data: bytes) -> dict:
    soup = BeautifulSoup(data, "html.parser")
    source_encoding = soup.original_encoding
    source_paths = original_paths(soup)
    removed = {}
    # Preserve original bytes separately; omit only non-reader content here.
    for tag in list(soup.find_all(["head", "script", "style", "ix:header", "ix:hidden"])):
        if tag.parent is not None:
            removed[tag.name] = removed.get(tag.name, 0) + 1
            tag.decompose()
    for tag in list(soup.find_all(True)):
        if tag.parent is not None and (tag.has_attr("hidden") or re.search(
                r"(?:display\s*:\s*none|visibility\s*:\s*hidden)", tag.get("style", ""), re.I)):
            removed["hidden_style_or_attribute"] = removed.get("hidden_style_or_attribute", 0) + 1
            tag.decompose()
    parts, spans, anchors, tag_ranges, paths = [], [], {}, {}, {}
    structures, images = [], []
    pos, last = 0, ""

    def emit(text):
        nonlocal pos, last
        if text:
            parts.append(text)
            pos += len(text)
            last = (last + text)[-2:]

    def newline():
        if pos and not last.endswith("\n"):
            emit("\n")

    def walk(node, path):
        if type(node) is NavigableString:
            value = re.sub(r"\s+", " ", str(node))
            if value.strip() or (pos and last[-1:] not in {" ", "\n", "\t"}):
                start = pos
                emit(value)
                spans.append({"start": start, "end": pos, "dom_path": path})
            return
        if not isinstance(node, Tag):
            return
        name = node.name
        paths[id(node)] = path
        if name in BLOCKS or name == "tr":
            newline()
        start = pos
        for attr in ("id", "name"):
            if node.get(attr):
                anchors.setdefault(str(node[attr]), {"offset": start, "dom_path": path})
        if name in {"br", "hr"}:
            newline()
        if name == "img":
            images.append({"src": node.get("src"), "alt": node.get("alt", ""),
                           "offset": pos, "dom_path": path, "downloaded": False})
            if node.get("alt"):
                emit("[image alt: " + str(node["alt"]) + "]")
        for child in node.children:
            if id(child) in source_paths:
                walk(child, source_paths[id(child)])
        end = pos
        tag_ranges[id(node)] = (start, end)
        if name in {"table", "tr", "td", "th"}:
            structures.append({"kind": name, "start": start, "end": end,
                               "dom_path": path, "rowspan": node.get("rowspan", "1"),
                               "colspan": node.get("colspan", "1")})
        if name in {"td", "th"}:
            emit("\t")
        elif name in BLOCKS or name == "tr":
            newline()

    walk(soup, "")
    text = "".join(parts)
    sections = [{"kind": "document", "title": "Full document", "start": 0,
                 "end": len(text), "dom_path": "/", "source_anchor": None,
                 "boundary_basis": "entire_readable_projection"}]
    candidates = {}
    for link in soup.find_all("a", href=True):
        href = str(link["href"])
        label = re.sub(r"\s+", " ", link.get_text(" ", strip=True))
        row = link.find_parent("tr")
        row_label = re.sub(r"\s+", " ", row.get_text(" ", strip=True)) if row else label
        if href.startswith("#") and unquote(href[1:]) in anchors:
            if ITEM.match(row_label) or re.fullmatch(r"SIGNATURES?", label, re.I):
                anchor = unquote(href[1:])
                target = anchors[anchor]
                # Duplicate TOC links at one target are one navigation address.
                candidates.setdefault(target["offset"], {
                    "kind": "item", "title": row_label[:400], "start": target["offset"],
                    "dom_path": target["dom_path"], "source_anchor": anchor,
                    "boundary_basis": "source_toc_anchor_to_next_toc_anchor"})
    ordered = sorted(candidates.values(), key=lambda x: x["start"])
    for i, section in enumerate(ordered):
        section["end"] = ordered[i + 1]["start"] if i + 1 < len(ordered) else len(text)
        sections.append(section)
    for tag in soup.find_all(re.compile(r"^h[1-6]$")):
        start, end = tag_ranges[id(tag)]
        sections.append({"kind": "heading", "title": tag.get_text(" ", strip=True)[:400],
                         "start": start, "end": end, "dom_path": paths[id(tag)],
                         "source_anchor": tag.get("id"), "boundary_basis": "heading_element_only"})
    tables = sorted((x for x in structures if x["kind"] == "table"), key=lambda x: x["start"])
    for i, table in enumerate(tables, 1):
        sections.append({"kind": "table", "title": f"HTML table {i} (may be layout)",
                         "start": table["start"], "end": table["end"],
                         "dom_path": table["dom_path"], "source_anchor": None,
                         "boundary_basis": "source_table_element"})
    return {"text": text, "spans": spans, "sections": sections,
            "structure": {"tables_and_cells": structures, "anchors": anchors, "images": images},
            "projection": {"encoding": source_encoding, "removed_elements": removed,
                           "text_characters": len(text), "text_nodes": len(spans),
                           "html_tables": len(tables), "image_count": len(images),
                           "no_item_filtering": True, "fixed_size_chunking": False,
                           "table_view": "row newlines and cell tabs; original cell spans preserved in HTML",
                           "external_images_and_exhibits_fetched": False,
                           "css_visual_rendering_and_ocr": False}}


SCHEMA = """
PRAGMA foreign_keys=ON;
CREATE TABLE store_metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE filings(
 filing_id TEXT PRIMARY KEY, accession TEXT NOT NULL UNIQUE, cik TEXT NOT NULL,
 ticker TEXT NOT NULL, company_name_at_metadata_snapshot TEXT NOT NULL,
 form TEXT NOT NULL, filed_at TEXT NOT NULL, accepted_at TEXT,
 report_date TEXT, is_amendment INTEGER NOT NULL, amends_filing_id TEXT,
 FOREIGN KEY(amends_filing_id) REFERENCES filings(filing_id));
CREATE TABLE documents(
 document_id TEXT PRIMARY KEY, filing_id TEXT NOT NULL REFERENCES filings(filing_id),
 role TEXT NOT NULL, original_filename TEXT NOT NULL, source_url TEXT NOT NULL,
 acquired_from_url TEXT NOT NULL, provenance TEXT NOT NULL,
 sec_bytes_verified INTEGER NOT NULL, html_path TEXT NOT NULL, text_path TEXT NOT NULL,
 html_sha256 TEXT NOT NULL, text_sha256 TEXT NOT NULL,
 text_characters INTEGER NOT NULL, original_metadata_path TEXT NOT NULL);
CREATE TABLE sections(
 section_id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(document_id),
 parent_section_id TEXT REFERENCES sections(section_id), kind TEXT NOT NULL,
 title TEXT NOT NULL, start_offset INTEGER NOT NULL, end_offset INTEGER NOT NULL,
 dom_path TEXT NOT NULL, source_anchor TEXT, boundary_basis TEXT NOT NULL);
CREATE TABLE text_spans(
 document_id TEXT NOT NULL REFERENCES documents(document_id), start_offset INTEGER NOT NULL,
 end_offset INTEGER NOT NULL, dom_path TEXT NOT NULL,
 PRIMARY KEY(document_id,start_offset,end_offset));
CREATE TABLE finrate_mapping(
 chunk_id TEXT PRIMARY KEY, title_filing_key TEXT NOT NULL,
 document_id TEXT NOT NULL REFERENCES documents(document_id), title_date TEXT NOT NULL,
 title_form TEXT NOT NULL, mapping_status TEXT NOT NULL, identity_basis TEXT NOT NULL,
 start_offset INTEGER, end_offset INTEGER,
 CHECK((start_offset IS NULL) = (end_offset IS NULL)));
CREATE INDEX filing_filter ON filings(ticker,filed_at,form);
CREATE INDEX section_lookup ON sections(document_id,start_offset,end_offset);
CREATE INDEX span_lookup ON text_spans(document_id,start_offset,end_offset);
"""


class RawStore:
    """Read-only tools. Character offsets use Python Unicode indexing, end-exclusive."""
    def __init__(self, root=DEFAULT_STORE):
        self.root = Path(root).resolve()
        self.db = sqlite3.connect((self.root / "manifest.sqlite").as_uri() + "?mode=ro", uri=True)
        self.db.row_factory = sqlite3.Row

    def close(self):
        self.db.close()

    def _document(self, document_id):
        row = self.db.execute("SELECT * FROM documents WHERE document_id=?", (document_id,)).fetchone()
        if row is None:
            raise ValueError("Unknown document_id")
        return dict(row)

    def _text(self, document):
        path = (self.root / document["text_path"]).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Document path outside store")
        return path.read_text(encoding="utf-8")

    def list_filings(self, ticker=None, form=None, filed_from=None, filed_to=None,
                     offset=0, limit=20, include_sections=False):
        if offset < 0 or not 1 <= limit <= 100:
            raise ValueError("offset >= 0; limit 1..100")
        terms, args = [], []
        for column, operator, value in (("f.ticker", "=", ticker), ("f.form", "=", form),
                                        ("f.filed_at", ">=", filed_from), ("f.filed_at", "<=", filed_to)):
            if value is not None:
                terms.append(f"{column}{operator}?")
                args.append(value)
        where = " WHERE " + " AND ".join(terms) if terms else ""
        query = " FROM filings f JOIN documents d USING(filing_id)" + where
        total = self.db.execute("SELECT count(*)" + query, args).fetchone()[0]
        rows = self.db.execute("SELECT f.*,d.document_id,d.text_characters,d.source_url,d.provenance" +
                               query + " ORDER BY f.ticker,f.filed_at,f.filing_id LIMIT ? OFFSET ?",
                               [*args, limit, offset]).fetchall()
        items = [dict(row) for row in rows]
        if include_sections:
            for item in items:
                item["sections"] = [dict(r) for r in self.db.execute(
                    "SELECT section_id,kind,title,start_offset,end_offset FROM sections "
                    "WHERE document_id=? AND kind IN ('item','heading') ORDER BY start_offset",
                    (item["document_id"],))]
        return {"total": total, "items": items,
                "next_offset": offset + len(items) if offset + len(items) < total else None}

    def read(self, document_id, start=0, length=12000, section_id=None):
        if start < 0 or not 1 <= length <= 24000:
            raise ValueError("start >= 0; length 1..24000")
        document = self._document(document_id)
        if section_id is not None:
            row = self.db.execute("SELECT start_offset FROM sections WHERE section_id=? AND document_id=?",
                                  (section_id, document_id)).fetchone()
            if row is None:
                raise ValueError("Unknown section for this document")
            start = row[0]
        text = self._text(document)
        if start > len(text):
            raise ValueError("start exceeds document length")
        end = min(len(text), start + length)
        first = self.db.execute("SELECT dom_path FROM text_spans WHERE document_id=? "
                                "AND end_offset>? AND start_offset<? ORDER BY start_offset LIMIT 1",
                                (document_id, start, end)).fetchone()
        return {"document_id": document_id, "start": start, "end": end,
                "total_characters": len(text), "text": text[start:end],
                "next_start": end if end < len(text) else None,
                "text_sha256": document["text_sha256"], "source_url": document["source_url"],
                "first_dom_path": first[0] if first else None}

    def grep(self, pattern, document_ids=None, regex=False, case_sensitive=False, offset=0, limit=20):
        if not pattern or offset < 0 or not 1 <= limit <= 100:
            raise ValueError("nonempty pattern; offset >= 0; limit 1..100")
        if regex:
            raise ValueError("Regex is not enabled in this storage smoke; use a literal pattern")
        matcher = re.compile(re.escape(pattern), 0 if case_sensitive else re.I)
        ids = document_ids if document_ids is not None else [r[0] for r in self.db.execute(
            "SELECT document_id FROM documents ORDER BY document_id")]
        matches, seen = [], 0
        for document_id in ids:
            text = self._text(self._document(document_id))
            for match in matcher.finditer(text):
                if seen >= offset:
                    if len(matches) == limit:
                        return {"matches": matches, "next_offset": offset + limit, "has_more": True}
                    lo, hi = max(0, match.start() - 180), min(len(text), match.end() + 180)
                    matches.append({"document_id": document_id, "match_start": match.start(),
                                    "match_end": match.end(), "start": lo, "end": hi, "text": text[lo:hi]})
                seen += 1
        return {"matches": matches, "next_offset": None, "has_more": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("tool", choices=["list_filings", "grep", "read"])
    parser.add_argument("arguments", nargs="?", default="{}", help="JSON tool arguments")
    args = parser.parse_args()
    store = RawStore(args.store)
    try:
        print(json.dumps(getattr(store, args.tool)(**json.loads(args.arguments)), ensure_ascii=False, indent=2))
    finally:
        store.close()


if __name__ == "__main__":
    main()
