"""Host-side detection of FIN-RATE benchmark exposure. No model calls; archives are never extracted.

Ported from finautorubric ``final_pipeline/contamination.py``; the detection rules are
unchanged except that web-search actions and declared answer sources count as agent
requests, a benchmark page the agent opened is hard evidence, and long unquoted copies
of the question in a search are advisory.
"""
from __future__ import annotations

import json
import re
import tarfile
from pathlib import Path

POLICY_VERSION = "sandbox-contamination/finrate-v1"

# Resolved to 0.0.0.0 inside every sandbox through /etc/hosts, so shell fetches
# (curl, wget, Python) fail. Codex's server-side web search is not covered by this
# list; trace detection below is the authoritative control.
BLOCKED_HOSTS = (
    "huggingface.co", "www.huggingface.co", "hf.co", "hf-mirror.com",
    "cdn-lfs.huggingface.co", "cdn-lfs.hf.co",
    "cdn-lfs-us-1.huggingface.co", "cdn-lfs-us-1.hf.co",
    "cdn-lfs-eu-1.huggingface.co", "cdn-lfs-eu-1.hf.co",
    "datasets-server.huggingface.co",
    "github.com", "www.github.com", "api.github.com", "codeload.github.com",
    "raw.githubusercontent.com", "objects.githubusercontent.com",
    "gist.github.com", "gist.githubusercontent.com",
    "arxiv.org", "www.arxiv.org", "export.arxiv.org",
)

# FIN-RATE identifiers (case-insensitive): dataset name, HF owner GGLabYale/Fin-RATE, arXiv id.
DATASET_MARKERS = ("fin-rate", "fin_rate", "finrate", "gglabyale", "2602.07294")
# Answer-key field names: blocking wherever they appear. FIN-RATE's own fields
# (answer, key_points, doc_ids) are too generic to use as markers.
ANSWER_KEY_MARKERS = ("reference_answer", "answer_key", "gold_answer")
_ANSWER_KEY_PATTERNS = tuple(
    (marker, re.compile(r"(?<![A-Za-z0-9_])" + re.escape(marker) + r"(?![A-Za-z0-9_])", re.IGNORECASE))
    for marker in ANSWER_KEY_MARKERS
)

# Strings the agent produced: commands, searches, opened URLs, file paths, declared sources.
REQUEST_KEYS = frozenset({"command", "cmd", "query", "queries", "file_path", "path", "pattern", "prompt",
                          "sources"})
# Parents whose every string is agent-produced: tool-call input, Codex web-search action.
REQUEST_PARENTS = frozenset({"input", "action"})
# Strings describing a page shown to the model, e.g. server-side search results.
SURFACE_KEYS = frozenset({"url", "domain", "title"})
SEARCH_KEYS = frozenset({"query", "queries"})
ARCHIVES = ("work.tar", "sessions.tar")
ARCHIVE_MEMBER_LIMIT = 64 * 1024**2
UNQUOTED_VERBATIM_WORDS = 10

_HOST_PATTERNS = tuple(
    (host, re.compile(r"(?<![A-Za-z0-9-])" + re.escape(host) + r"(?![A-Za-z0-9-])", re.IGNORECASE))
    for host in BLOCKED_HOSTS
)
_URL = re.compile(r"https?://([A-Za-z0-9.-]+)(/[^\s\"'<>)\]]*)?", re.IGNORECASE)


def hosts_in(text: str) -> list:
    """Blocked hosts referenced in text, including their subdomains."""
    return sorted({host for host, pattern in _HOST_PATTERNS if pattern.search(text)})


def dataset_markers_in(text: str) -> list:
    low = text.lower()
    return sorted({marker for marker in DATASET_MARKERS if marker in low})


def answer_keys_in(text: str) -> list:
    return sorted({marker for marker, pattern in _ANSWER_KEY_PATTERNS if pattern.search(text)})


def markers_in(text: str) -> list:
    return sorted(set(dataset_markers_in(text)) | set(answer_keys_in(text)))


def benchmark_urls_in(text: str) -> list:
    """Blocked-host URLs that point at a dataset, not an incidental code-hosting link."""
    found = set()
    for match in _URL.finditer(text):
        host, path = match.group(1), match.group(2) or ""
        if hosts_in(host) and (dataset_markers_in(path) or "/datasets/" in path.lower()):
            found.update(hosts_in(host))
    return sorted(found)


def _normal(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9%$.,'-]+", text.lower()))


def verbatim_spans(query: str, raw_query: str, min_words: int = 5) -> list:
    """Spans of the original question used as a search string (advisory).

    A quoted span of ``min_words`` words, or any unquoted run of
    ``UNQUOTED_VERBATIM_WORDS`` consecutive words, copied from the question.
    """
    target = _normal(raw_query)
    spans = []
    for quoted in re.findall(r'"([^"]{10,})"', query):
        normal = _normal(quoted)
        if len(normal.split()) >= min_words and normal in target:
            spans.append(quoted)
    words = re.findall(r"[a-z0-9]+", query.lower())
    question = " " + " ".join(re.findall(r"[a-z0-9]+", raw_query.lower())) + " "
    for start in range(len(words) - UNQUOTED_VERBATIM_WORDS + 1):
        window = " ".join(words[start:start + UNQUOTED_VERBATIM_WORDS])
        if " " + window + " " in question:
            spans.append(window)
            break
    return spans


def opened_pages(value):
    """Yield (url, title) for every page the agent opened.

    A Codex web-search item whose action is not a search (open, click, find) carries the
    viewed page in ``results``; older rollouts record ``{"type": "open_page", "url": ...}``.
    """
    if isinstance(value, dict):
        action = value.get("action")
        if isinstance(action, dict) and action.get("type") not in (None, "search"):
            if action.get("type") == "open_page" and isinstance(action.get("url"), str):
                yield action["url"], ""
            for result in value.get("results") or []:
                if isinstance(result, dict) and isinstance(result.get("url"), str):
                    yield result["url"], str(result.get("title") or "")
        for child in value.values():
            yield from opened_pages(child)
    elif isinstance(value, list):
        for child in value:
            yield from opened_pages(child)


def _strings(value, key=None, parent=None):
    """Yield (key, parent_key, text) for string values and ("__key__", parent, name) for dict keys."""
    if isinstance(value, dict):
        for child_key, child in value.items():
            yield "__key__", key, str(child_key)
            yield from _strings(child, child_key, key)
    elif isinstance(value, list):
        for child in value:
            yield from _strings(child, key, parent)
    elif isinstance(value, str):
        yield key, parent, value


class _Collector:
    def __init__(self, source: str, raw_query: str = ""):
        self.source, self.raw_query = source, raw_query
        self.evidence, self.advisory, self._seen = [], [], set()

    def _add(self, bucket, entry):
        token = json.dumps(entry, sort_keys=True)
        if token not in self._seen:
            self._seen.add(token)
            bucket.append(entry)

    def string(self, key, text, parent=None):
        # Evidence never carries fetched content: it may contain an answer key.
        if key == "__key__":
            # A JSON field named like an answer key (e.g. a downloaded dataset record).
            for match in answer_keys_in(text):
                self._add(self.evidence, {"kind": "content", "source": self.source, "field": "key", "match": match})
            for match in dataset_markers_in(text):
                self._add(self.advisory, {"kind": "dataset_surfaced", "source": self.source, "field": "key", "match": match})
            return
        if key in REQUEST_KEYS or parent in REQUEST_PARENTS:
            hard = markers_in(text) + benchmark_urls_in(text)
            for match in hosts_in(text):
                if match not in hard:
                    self._add(self.advisory, {"kind": "blocked_host_requested", "source": self.source,
                                              "field": key, "match": match})
            for match in hard:
                self._add(self.evidence, {"kind": "request", "source": self.source, "field": key,
                                          "match": match, "detail": text[:200]})
            if key in SEARCH_KEYS and self.raw_query:
                for span in verbatim_spans(text, self.raw_query):
                    self._add(self.advisory, {"kind": "verbatim_query_search", "source": self.source,
                                              "detail": span[:200]})
        elif key in SURFACE_KEYS:
            for match in hosts_in(text) + markers_in(text):
                self._add(self.advisory, {"kind": "search_result_surfaced", "source": self.source,
                                          "field": key, "match": match, "detail": text[:200]})
        else:
            if len(text) <= 4 * 1024**2 and text.lstrip().startswith(("{", "[")):
                try:
                    nested = json.loads(text)
                except ValueError:
                    nested = None
                if isinstance(nested, (dict, list)):
                    for child_key, child_parent, value in _strings(nested):
                        self.string(child_key, value, child_parent)
            self.text(text, field=key or "")

    def text(self, text, field=""):
        for match in answer_keys_in(text):
            self._add(self.advisory, {"kind": "answer_marker_surfaced", "source": self.source,
                                      "field": field, "match": match})
        for match in benchmark_urls_in(text):
            self._add(self.advisory, {"kind": "benchmark_url_surfaced", "source": self.source,
                                      "field": field, "match": match})
        for match in dataset_markers_in(text):
            self._add(self.advisory, {"kind": "dataset_surfaced", "source": self.source, "field": field, "match": match})

    def jsonl(self, text):
        """Walk JSON lines; non-JSON lines fall back to plain text rules."""
        for line in text.splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                self.text(line)
                continue
            if isinstance(event, (dict, list)):
                for key, parent, value in _strings(event):
                    self.string(key, value, parent)
                # A page the agent chose to open is exposure, not a mere search hit.
                for url, title in opened_pages(event):
                    for match in markers_in(url + " " + title) + benchmark_urls_in(url):
                        self._add(self.evidence, {"kind": "page_opened", "source": self.source,
                                                  "match": match, "detail": url[:200]})
            elif isinstance(event, str):
                self.text(event)

    def json_document(self, text):
        """Walk one JSON document, including pretty-printed work artifacts."""
        try:
            value = json.loads(text)
        except ValueError:
            self.text(text)
            return
        if isinstance(value, (dict, list)):
            for key, parent, item in _strings(value):
                self.string(key, item, parent)
        elif isinstance(value, str):
            self.text(value)


def scan_trace(path: Path, raw_query: str = ""):
    """Format-agnostic scan of a Codex JSONL trace."""
    collector = _Collector(path.name, raw_query)
    collector.jsonl(path.read_text(errors="replace"))
    return collector.evidence, collector.advisory


def scan_archive(path: Path, raw_query: str = "", limit: int = ARCHIVE_MEMBER_LIMIT):
    """Inspect member names and regular-file bytes in memory; nothing is extracted to disk."""
    collector = _Collector(path.name, raw_query)
    if not path.exists():
        return collector.evidence, collector.advisory
    try:
        with tarfile.open(path, "r:") as archive:
            for member in archive:
                for match in dataset_markers_in(member.name) + answer_keys_in(member.name):
                    collector._add(collector.evidence, {"kind": "archive_name", "source": path.name,
                                                        "match": match, "detail": member.name[:200]})
                if not member.isfile():
                    continue
                if member.size > limit:
                    collector._add(collector.advisory, {"kind": "archive_member_unscanned", "source": path.name,
                                                        "detail": member.name[:200]})
                    continue
                handle = archive.extractfile(member)
                text = handle.read().decode("utf-8", errors="replace") if handle else ""
                collector.source = path.name + ":" + member.name[:120]
                if member.name.endswith(".jsonl"):
                    collector.jsonl(text)
                elif member.name.endswith(".json"):
                    collector.json_document(text)
                else:
                    collector.text(text)
                collector.source = path.name
    except tarfile.TarError as exc:
        collector.advisory.append({"kind": "archive_unreadable", "source": path.name, "detail": str(exc)[:200]})
    return collector.evidence, collector.advisory


def detect(directory: Path, raw_query: str = "") -> dict:
    """Scan the Codex event traces, the /work archive and the Codex session rollout archive."""
    evidence, advisory, scanned = [], [], []
    for trace in sorted(directory.glob("turn_*.jsonl")):
        found, notes = scan_trace(trace, raw_query)
        evidence.extend(found)
        advisory.extend(notes)
        scanned.append(trace.name)
    for name in ARCHIVES:
        if (directory / name).exists():
            found, notes = scan_archive(directory / name, raw_query)
            evidence.extend(found)
            advisory.extend(notes)
            scanned.append(name)
    summary = sorted({item["kind"] + ":" + item["match"] for item in evidence})
    advisory_summary = sorted({item["kind"] + ":" + item.get("match", "") for item in advisory})
    status = "blocked" if evidence else "suspect" if advisory else "clean"
    return {"policy": POLICY_VERSION, "status": status, "detected": bool(evidence), "scanned": scanned,
            "summary": summary, "advisory_summary": advisory_summary,
            "evidence": evidence[:50], "advisory": advisory[:20]}
