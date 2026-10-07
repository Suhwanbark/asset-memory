"""Append-only, source-grounded fact histories with bounded agent tools.

An updated observation does not make the earlier observation historically false.
Missing mentions never imply termination. Tool results are evidence, not commands.
"""
from __future__ import annotations

from datetime import date
import json
from pathlib import Path
import re
import sqlite3

from .runtime import check_disk, digest

FACT_FIELDS = {"subject", "topic", "predicate", "value", "period", "modality",
               "source_id", "quote"}
MODALITIES = {"reported", "risk", "forecast", "plan", "explicit_retraction"}
RELATIONS = {"continues", "updates", "contradicts", "retracts", "same_topic"}


def source_span(text, proposed_quote):
    """Align whitespace only; never change numbers, words, case or punctuation."""
    if not isinstance(proposed_quote, str) or len(proposed_quote.strip()) < 12:
        raise ValueError("Quote is missing or too short")
    at = text.find(proposed_quote)
    if at >= 0:
        return at, at+len(proposed_quote), "exact"
    pieces = proposed_quote.split()
    pattern = r"\s+".join(re.escape(piece) for piece in pieces)
    match = re.search(pattern, text)
    if not match:
        raise ValueError("Quote differs from source beyond whitespace")
    return match.start(), match.end(), "whitespace_normalized"


def canonical(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class Bank:
    def __init__(self, path, *, readonly=False):
        self.path = Path(path)
        self.readonly = readonly
        if readonly:
            self.db = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        else:
            check_disk()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.db = sqlite3.connect(self.path)
            self.db.executescript("""
                PRAGMA foreign_keys=ON;
                CREATE TABLE IF NOT EXISTS sources (
                    id TEXT PRIMARY KEY, published TEXT NOT NULL, title TEXT NOT NULL,
                    text TEXT NOT NULL, sha256 TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS facts (
                    id TEXT PRIMARY KEY, source_id TEXT NOT NULL REFERENCES sources(id),
                    observed_at TEXT NOT NULL, body TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS links (
                    id TEXT PRIMARY KEY, old_id TEXT NOT NULL REFERENCES facts(id),
                    new_id TEXT NOT NULL REFERENCES facts(id), body TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, body TEXT NOT NULL,
                    previous_hash TEXT NOT NULL, sha256 TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS checkpoints (
                    label TEXT PRIMARY KEY, event_hash TEXT NOT NULL, body TEXT NOT NULL);
            """)
        self.db.row_factory = sqlite3.Row

    def close(self):
        self.db.close()

    def _writable(self):
        if self.readonly:
            raise PermissionError("Evaluation cannot mutate the shared bank")
        check_disk()

    def _event(self, body):
        previous = self.db.execute("SELECT sha256 FROM events ORDER BY seq DESC LIMIT 1").fetchone()
        previous_hash = previous[0] if previous else "0" * 64
        content = canonical(body)
        event_hash = digest(previous_hash + content)
        self.db.execute("INSERT INTO events(body,previous_hash,sha256) VALUES(?,?,?)",
                        (content, previous_hash, event_hash))
        return event_hash

    def register_source(self, source):
        self._writable()
        published = date.fromisoformat(source["published"]).isoformat()
        payload = (source["id"], published, source["title"], source["text"], digest(source["text"]))
        old = self.db.execute("SELECT * FROM sources WHERE id=?", (source["id"],)).fetchone()
        if old:
            if tuple(old) != payload:
                raise ValueError("A source ID cannot be overwritten")
            return
        latest = self.db.execute("SELECT MAX(published) FROM sources").fetchone()[0]
        if latest and published < latest:
            raise ValueError("Sources must be ingested chronologically; rebuild for late arrivals")
        with self.db:
            self.db.execute("INSERT INTO sources VALUES(?,?,?,?,?)", payload)
            self._event({"op": "register_source", "id": source["id"],
                         "published": published, "text_sha256": payload[-1]})

    def read_source(self, source_id, start=0, length=6000):
        if start < 0 or not 1 <= length <= 20000:
            raise ValueError("Invalid source window")
        row = self.db.execute("SELECT * FROM sources WHERE id=?", (source_id,)).fetchone()
        if not row:
            raise KeyError(source_id)
        return {"id": source_id, "title": row["title"], "published": row["published"],
                "start": start, "text": row["text"][start:start+length], "sha256": row["sha256"]}

    def add_fact(self, fact):
        self._writable()
        if set(fact) != FACT_FIELDS:
            raise ValueError(f"Fact schema mismatch: {set(fact) ^ FACT_FIELDS}")
        if fact["modality"] not in MODALITIES:
            raise ValueError("Invalid modality")
        if not all(isinstance(v, str) and v.strip() for k, v in fact.items() if k != "period"):
            raise ValueError("Empty fact field")
        # Unknown reporting periods stay null rather than becoming publication dates.
        if fact["period"] is not None and not isinstance(fact["period"], str):
            raise ValueError("period must be an explicit source-supported string or null")
        row = self.db.execute("SELECT * FROM sources WHERE id=?", (fact["source_id"],)).fetchone()
        if not row:
            raise ValueError("Unknown source")
        start, end, alignment = source_span(row["text"], fact["quote"])
        body = {**fact, "quote": row["text"][start:end], "quote_start": start,
                "quote_end": end, "extraction_quote": fact["quote"], "quote_alignment": alignment,
                "observed_at": row["published"], "source_sha256": row["sha256"]}
        fact_id = "fact_" + digest(canonical(body))[:24]
        if self.db.execute("SELECT 1 FROM facts WHERE id=?", (fact_id,)).fetchone():
            return fact_id
        with self.db:
            self.db.execute("INSERT INTO facts VALUES(?,?,?,?)", (
                fact_id, fact["source_id"], row["published"], canonical(body)))
            self._event({"op": "add_fact", "fact_id": fact_id, "body": body})
        return fact_id

    def read_fact(self, fact_id):
        row = self.db.execute("SELECT body FROM facts WHERE id=?", (fact_id,)).fetchone()
        if not row:
            raise KeyError(fact_id)
        return {"id": fact_id, **json.loads(row["body"])}

    def grep(self, pattern, *, subject=None, before=None, limit=20):
        """Literal, case-insensitive grep; no shell execution or arbitrary regex."""
        if not isinstance(pattern, str) or not pattern or len(pattern) > 200:
            raise ValueError("Invalid literal pattern")
        if not 1 <= limit <= 50:
            raise ValueError("limit must be 1..50")
        if before:
            date.fromisoformat(before)
        result = []
        for row in self.db.execute("SELECT id,body FROM facts ORDER BY observed_at DESC,id"):
            body = json.loads(row["body"])
            if subject and body["subject"].casefold() != subject.casefold():
                continue
            if before and body["observed_at"] > before:
                continue
            if pattern.casefold() in canonical(body).casefold():
                result.append({"id": row["id"], **body})
            if len(result) == limit:
                break
        return result

    def link(self, *, old_id, new_id, relation, rationale, evidence_quote):
        self._writable()
        if relation not in RELATIONS or old_id == new_id:
            raise ValueError("Invalid history link")
        old, new = self.read_fact(old_id), self.read_fact(new_id)
        if old["subject"].casefold() != new["subject"].casefold():
            raise ValueError("Cross-entity history requires separately verified entity alignment")
        if old["observed_at"] > new["observed_at"]:
            raise ValueError("History link runs backwards")
        source = self.db.execute("SELECT text FROM sources WHERE id=?", (new["source_id"],)).fetchone()[0]
        if not rationale:
            raise ValueError("Link requires a source-grounded rationale")
        start, end, alignment = source_span(source, evidence_quote)
        if relation == "retracts" and new["modality"] != "explicit_retraction":
            raise ValueError("Only explicit source retractions can retract older claims")
        body = {"old_id": old_id, "new_id": new_id, "relation": relation,
                "rationale": rationale, "evidence_quote": source[start:end],
                "proposed_evidence_quote": evidence_quote, "quote_alignment": alignment,
                "source_id": new["source_id"], "observed_at": new["observed_at"]}
        link_id = "link_" + digest(canonical(body))[:24]
        with self.db:
            if not self.db.execute("SELECT 1 FROM links WHERE id=?", (link_id,)).fetchone():
                self.db.execute("INSERT INTO links VALUES(?,?,?,?)", (link_id, old_id, new_id, canonical(body)))
                self._event({"op": "link", "link_id": link_id, "body": body})
        return link_id

    def history(self, fact_id, *, limit=20):
        self.read_fact(fact_id)
        if not 1 <= limit <= 50:
            raise ValueError("Invalid limit")
        seen = {fact_id}
        links = {}
        frontier = [fact_id]
        while frontier and len(links) < limit:
            current = frontier.pop(0)
            for row in self.db.execute("SELECT id,body FROM links WHERE old_id=? OR new_id=? ORDER BY id", (current, current)):
                link = json.loads(row["body"])
                links[row["id"]] = link
                for neighbor in (link["old_id"], link["new_id"]):
                    if neighbor not in seen:
                        seen.add(neighbor)
                        frontier.append(neighbor)
                if len(links) >= limit:
                    break
        return {"facts": sorted([self.read_fact(i) for i in seen], key=lambda f: (f["observed_at"], f["id"])),
                "links": list(links.values())}

    def checkpoint(self, quarter):
        self._writable()
        import re
        if not re.fullmatch(r"20\d{2}-Q[1-4]", quarter):
            raise ValueError("Expected YYYY-Qn")
        last = self.db.execute("SELECT sha256 FROM events ORDER BY seq DESC LIMIT 1").fetchone()
        event_hash = last[0] if last else "0" * 64
        body = {"quarter": quarter, "event_hash": event_hash,
                "sources": self.db.execute("SELECT COUNT(*) FROM sources").fetchone()[0],
                "facts": self.db.execute("SELECT COUNT(*) FROM facts").fetchone()[0],
                "links": self.db.execute("SELECT COUNT(*) FROM links").fetchone()[0]}
        old = self.db.execute("SELECT body FROM checkpoints WHERE label=?", (quarter,)).fetchone()
        if old and old[0] != canonical(body):
            raise ValueError("A quarter checkpoint is immutable")
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO checkpoints VALUES(?,?,?)", (quarter, event_hash, canonical(body)))
        return body

    def verify_event_chain(self):
        previous = "0" * 64
        for row in self.db.execute("SELECT * FROM events ORDER BY seq"):
            if row["previous_hash"] != previous or row["sha256"] != digest(previous + row["body"]):
                raise ValueError("Bank event chain was modified")
            previous = row["sha256"]
        return previous

    def tool(self, name, arguments, *, allow_edits=False):
        readers = {"grep": self.grep, "read_fact": self.read_fact,
                   "read_source": self.read_source, "history": self.history}
        writers = {"add_fact": self.add_fact, "link": self.link}
        if name in readers:
            return readers[name](**arguments)
        if name in writers and allow_edits:
            return writers[name](**arguments)
        raise PermissionError(f"Tool is not allowed in this phase: {name}")
