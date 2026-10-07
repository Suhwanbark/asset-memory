"""Compare deterministic corpus samples with accession-matched public filing copies.

SEC submissions metadata is authoritative for identity. The publicly accessible
Yahoo Finance copy supplies HTML when direct SEC Archives download returns 403.
Text-window matches detect omissions; they are not a proof of table/image fidelity.
No benchmark questions, model calls, or experiment processes are used.
"""
import csv
import hashlib
import json
import re
import shutil
import time
import unicodedata
import urllib.request
import zipfile
from collections import defaultdict
from datetime import datetime, timezone, timedelta, date as date_type
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/finrate_source_audit_20261007'
OUT = ROOT / 'result/finrate_source_audit_20261007'
CIKS = {'GEN': 849399, 'HUSA': 1156041, 'LANC': 57515, 'NXPL': 1058307,
        'REEMF': 1419806, 'GS': 886982, 'VLO': 1035002, 'HSBC': 1089113}
TITLE = re.compile(r'^(.*?)_(\d{4}-\d{2}-\d{2})_(.*?)_(\d+)_(.*)$')


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def fetch(url, path):
    if path.exists():
        return path.read_bytes()
    if shutil.disk_usage(ROOT).free < 10 * 1024**3:
        raise SystemExit('STOP: free disk below 10 GiB')
    time.sleep(0.3)
    req = urllib.request.Request(url, headers={'User-Agent': 'FinRATE-public-corpus-audit'})
    with urllib.request.urlopen(req, timeout=30) as response:
        content = response.read()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return content


def normalize(text):
    return ''.join(c for c in unicodedata.normalize('NFKC', text).casefold() if c.isalnum())


def form_label(form):
    return form.split('/')[0].split(' ')[0]


def source_index(content):
    soup = BeautifulSoup(content, 'html.parser')
    image_count = len(soup.find_all('img'))
    for node in list(soup.find_all(['script', 'style', 'head', 'ix:header', 'ix:hidden'])):
        if node.parent:
            node.decompose()
    for node in list(soup.find_all(style=True)):
        if node.parent and re.search(r'display\s*:\s*none', node.get('style', ''), re.I):
            node.decompose()
    parts, anchors, tag_positions = [], {}, {}
    length = 0
    for node in soup.descendants:
        if isinstance(node, Tag):
            tag_positions[id(node)] = length
            for attr in ['id', 'name']:
                if node.get(attr):
                    anchors[node[attr]] = length
        elif type(node) is NavigableString:  # Excludes comments and processing instructions.
            text = normalize(str(node))
            parts.append(text)
            length += len(text)
    starts = set()
    for link in soup.find_all('a', href=True):
        href = link['href']
        if not href.startswith('#') or href[1:] not in anchors:
            continue
        row = link.find_parent('tr')
        text = row.get_text(' ', strip=True) if row else link.get_text(' ', strip=True)
        match = re.match(r'^Item\s+(\d+(?:\.\d+)?[A-Z]?)\b', text, re.I)
        if match and len(text) < 600:
            starts.add((anchors[href[1:]], 'Item ' + match.group(1).upper()))
        if re.sub(r'\s+', '', link.get_text()).upper() == 'SIGNATURES':
            starts.add((anchors[href[1:]], 'SIGNATURES_AND_TAIL'))
    # Only use the source TOC anchors. Do not invent section boundaries if absent.
    return ''.join(parts), sorted(starts), image_count


def compare_windows(source, corpus, step=100):
    hits, misses, run_start = 0, [], None
    total = (len(source) + step - 1) // step
    for start in range(0, len(source), step):
        if source[start:start + step] in corpus:
            hits += 1
            if run_start is not None:
                misses.append((run_start, start))
                run_start = None
        elif run_start is None:
            run_start = start
    if run_start is not None:
        misses.append((run_start, len(source)))
    return {'windows': total, 'matched_windows': hits,
            'matched_percent': round(hits / total * 100, 4) if total else None,
            'long_unmatched_runs': [{'start': a, 'end': b, 'characters': b-a,
                                     'normalized_excerpt': source[a:min(a+200, b)]}
                                    for a, b in misses if b-a >= 500],
            'max_unmatched_run_chars': max((b-a for a, b in misses), default=0)}


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ROOT / 'data/finrate_public_v1/corpus/corpus.zip') as archive:
        with archive.open('corpus.jsonl') as f:
            records = [json.loads(line) for line in f]
    groups = defaultdict(list)
    for record in records:
        ticker, date, form, sequence, _ = TITLE.fullmatch(record['title']).groups()
        groups[(ticker, date, form)].append((int(sequence), record))
    selected = set()
    for ticker in ['GEN', 'HUSA', 'LANC', 'NXPL', 'REEMF', 'GS', 'VLO']:
        keys = sorted(k for k in groups if k[0] == ticker and k[2] == '10-K')
        selected.update([keys[0], keys[-1]])
    selected.update([('HUSA', '2022-03-31', '10-K'), ('GS', '2024-02-27', '10-K'),
                     ('REEMF', '2022-04-27', '10-K')])
    for ticker, form in [('HUSA', '10-Q'), ('VLO', '10-Q'), ('VLO', '8-K'),
                         ('REEMF', '8-K'), ('HUSA', 'DEF'), ('HSBC', '6-K')]:
        keys = sorted(k for k in groups if k[0] == ticker and k[2] == form)
        if keys:
            selected.add(keys[-1])
    plan = {'selection_rule': 'Earliest/latest title-labelled 10-K for five PoC companies plus GS/VLO; HUSA 2022 continuity example; GS/REEMF short 10-K examples; latest specified non-10-K forms.',
            'not_random_or_exhaustive': True, 'filings': [list(k) for k in sorted(selected)]}
    dump(OUT / 'selection.json', plan)
    metadata = {}
    for ticker in sorted({k[0] for k in selected}):
        cik = CIKS[ticker]
        main_name = f'CIK{cik:010d}.json'
        main = json.loads(fetch('https://data.sec.gov/submissions/' + main_name,
                                DATA / 'submissions' / main_name))
        tables = [main['filings']['recent']]
        dates = {k[1] for k in selected if k[0] == ticker}
        nearby_dates = {(date_type.fromisoformat(d) + timedelta(days=offset)).isoformat()
                        for d in dates for offset in range(-3, 4)}
        for history in main['filings'].get('files', []):
            if any(history['filingFrom'] <= date <= history['filingTo'] for date in nearby_dates):
                tables.append(json.loads(fetch('https://data.sec.gov/submissions/' + history['name'],
                                               DATA / 'submissions' / history['name'])))
        flattened = []
        for table in tables:
            for i, date in enumerate(table['filingDate']):
                if date in nearby_dates:
                    flattened.append({field: table[field][i] for field in
                                      ['filingDate', 'reportDate', 'acceptanceDateTime', 'form', 'accessionNumber', 'primaryDocument']})
        metadata[ticker] = flattened
    results, item_rows = [], []
    for ticker, date, form in sorted(selected):
        key = f'{ticker}_{date}_{form}'
        rows = sorted(groups[(ticker, date, form)], key=lambda x: (x[0], x[1]['_id']))
        matches = [r for r in metadata[ticker] if r['filingDate'] == date and form_label(r['form']) == form]
        match_basis = 'exact_title_date_and_normalized_form'
        if not matches:
            matches = [r for r in metadata[ticker] if form_label(r['form']) == form and
                       abs((date_type.fromisoformat(r['filingDate']) - date_type.fromisoformat(date)).days) <= 3]
            match_basis = 'unique_candidate_within_3_days_and_normalized_form; content_comparison_required'
        result = {'filing_key': key, 'ticker': ticker, 'title_date': date, 'title_form': form,
                  'chunk_count': len(rows), 'metadata_candidates': matches, 'identity_matching_basis': match_basis}
        if len(matches) != 1:
            result['status'] = 'ambiguous_metadata' if matches else 'unmatched_metadata'
            results.append(result)
            print(key, result['status'], len(matches), flush=True)
            continue
        match = matches[0]
        cik = CIKS[ticker]
        accession = match['accessionNumber'].replace('-', '')
        sec_url = f'https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{match["primaryDocument"]}'
        copy_url = f'https://cdn.yahoofinance.com/prod/sec-filings/{cik:010d}/{accession}/{match["primaryDocument"]}'
        result.update({'actual_sec_form': match['form'], 'accession': match['accessionNumber'],
                       'report_date': match['reportDate'], 'original_sec_url': sec_url,
                       'html_download_url': copy_url, 'copy_provenance': 'Public Yahoo Finance copy at matching SEC accession; not byte-verified against SEC download'})
        try:
            content = fetch(copy_url, DATA / (key + '.mirror.html'))
            result['html_sha256'] = hashlib.sha256(content).hexdigest()
            source, boundaries, image_count = source_index(content)
            if len(source) < 30:
                raise ValueError('Insufficient HTML text')
            markdown = '\n'.join(r['text'] for _, r in rows)
            markdown = re.sub(r'!\[[^\]]*\]\([^)]*\)', '', markdown)
            markdown = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', markdown)
            corpus = normalize(markdown)
            result.update({'status': 'compared', 'source_normalized_chars': len(source),
                           'corpus_normalized_chars': len(corpus), 'source_images_not_audited': image_count,
                           'full_primary_document': compare_windows(source, corpus),
                           'toc_item_boundaries_found': len([b for b in boundaries if b[1].startswith('Item ')])})
            for i, (start, label) in enumerate(boundaries):
                end = boundaries[i+1][0] if i+1 < len(boundaries) else len(source)
                if start >= end:
                    continue
                text = source[start:end]
                measured = compare_windows(text, corpus)
                item_rows.append({'filing_key': key, 'source_section': label,
                                  'source_start': start, 'source_end': end, 'source_normalized_chars': len(text),
                                  **{k: v for k, v in measured.items() if k != 'long_unmatched_runs'}})
            dump(DATA / (key + '.meta.json'), {k: v for k, v in result.items() if k != 'full_primary_document'})
            print(key, match['form'], result['full_primary_document']['matched_percent'],
                  'max_gap', result['full_primary_document']['max_unmatched_run_chars'], flush=True)
        except Exception as error:
            result.update({'status': 'fetch_or_parse_failed', 'error': str(error)})
            print(key, result['status'], str(error), flush=True)
        results.append(result)
        dump(OUT / 'filing_comparisons.json', results)
    dump(OUT / 'filing_comparisons.json', results)
    dump(OUT / 'section_comparisons.json', item_rows)
    if item_rows:
        with (OUT / 'section_comparisons.csv').open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=list(item_rows[0]))
            writer.writeheader()
            writer.writerows(item_rows)
    summary = {'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'selected': len(selected),
               'compared': sum(r['status'] == 'compared' for r in results),
               'corpus_revision': '284b1b4f6cd1e58574f1f1d17c29b35c6aa72cce',
               'method': 'Visible HTML text (excluding comments/script/style/hidden XBRL), NFKC/casefold/alphanumeric normalization, exact search of each consecutive 100-character source window in concatenated same-filing corpus text.',
               'scope': 'Primary HTML document only; not images, linked exhibits, incorporated-by-reference filings, table layout or numeric sign fidelity.',
               'limitations': ['Deterministic diagnostic sample, not all 2472 filings.',
                               'Metadata identity comes from SEC; HTML is a public mirror copy.',
                               'Exact windows can fail at removed boilerplate or Markdown representation changes.',
                               'Text present elsewhere in the same filing can match, so matching does not prove original order or location.',
                               'Intentional omission cannot be established without preprocessing code or author confirmation.']}
    dump(OUT / 'summary.json', summary)
    print('SUMMARY', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
