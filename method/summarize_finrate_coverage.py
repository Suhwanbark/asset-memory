"""Report corpus coverage and confirmed source omissions with explicit denominators."""
import csv
import json
import re
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from bs4 import BeautifulSoup
from audit_finrate_source_sections import source_index, normalize

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'result/finrate_coverage_statistics_20261007'


def ratio(numerator, denominator):
    return {'numerator': numerator, 'denominator': denominator,
            'percent': round(numerator / denominator * 100, 4)}


def main():
    inventory = json.loads((ROOT / 'result/finrate_corpus_inventory_20261007/summary.json').read_text())
    with (ROOT / 'result/finrate_corpus_inventory_20261007/companies.csv').open(encoding='utf-8-sig') as f:
        companies = list(csv.DictReader(f))
    with zipfile.ZipFile(ROOT / 'data/finrate_public_v1/corpus/corpus.zip') as archive:
        with archive.open('corpus.jsonl') as f:
            documents = [json.loads(line) for line in f]
    expression = re.compile(r'^(.*?)_(\d{4}-\d{2}-\d{2})_(.*?)_(\d+)_(.*)$')
    doc_to_filing = {r['_id']: expression.fullmatch(r['title']).groups()[:3] for r in documents}
    task_coverage, all_refs = [], set()
    for task in ['DR-QA', 'LT-QA', 'EC-QA']:
        rows = json.loads((ROOT / 'data/finrate_public_v1/qa' / (task + '.json')).read_text())
        refs = set()
        for row in rows:
            refs.update(row.get('doc_ids', [row.get('doc_id')]))
        assert refs <= doc_to_filing.keys()
        all_refs.update(refs)
        task_coverage.append({'task': task, 'questions': len(rows), 'gold_referenced_chunks': len(refs),
                              'gold_referenced_filing_groups': len({doc_to_filing[x] for x in refs}),
                              'companies_with_gold_evidence': len({doc_to_filing[x][0] for x in refs})})
    company_years = sum(sum(int(row[f'filings_{year}']) > 0 for year in range(2020, 2026)) for row in companies)
    assert company_years == sum(row['companies'] for row in inventory['year_counts'])
    assert sum(row['chunks'] for row in inventory['year_counts']) == len(documents)
    assert sum(row['filing_groups'] for row in inventory['year_counts']) == inventory['filing_groups_by_ticker_date_form']
    source_dir = ROOT / 'data/finrate_source_audit_20261007'
    cases = []
    for key, omitted, expected_total, phrase in [
        ('NXPL_2023-03-31_10-K', ['11', '12', '13', '14'], 22,
         'The information required by this Item will be included in the 2023 Proxy Statement and is incorporated herein by reference.'),
        ('REEMF_2024-10-25_8-K', ['2.01', '3.02'], 3,
         'The information set forth in Item 8.01 is incorporated herein by reference.')]:
        content = (source_dir / (key + '.mirror.html')).read_bytes()
        source, boundaries, _ = source_index(content)
        soup = BeautifulSoup(content, 'html.parser')
        body_labels = set()
        for tag in soup.find_all('p'):
            text = re.sub(r'\s+', ' ', tag.get_text(' ', strip=True))
            match = re.match(r'^Item\s+(\d+(?:\.\d+)?[A-Z]?)\b', text, re.I)
            if match and len(text) <= 240 and not any(a.get('href', '').startswith('#') for a in tag.find_all('a', href=True)):
                body_labels.add(match.group(1).upper())
        assert len(body_labels) == expected_total, (key, sorted(body_labels))
        rows = [r for r in documents if r['title'].startswith(key + '_')]
        corpus = normalize('\n'.join(r['text'] for r in rows))
        assert source.count(normalize(phrase)) == len(omitted)
        assert corpus.count(normalize(phrase)) == 0
        if key.startswith('REEMF'):
            boundaries = [(source.index(normalize(label)), label) for label in
                          ['Item 2.01', 'Item 3.02', 'Item 8.01 Other Events']]
        missing_spans = []
        for i, (start, label) in enumerate(boundaries):
            if label not in ['Item ' + item for item in omitted]:
                continue
            end = boundaries[i+1][0] if i+1 < len(boundaries) else len(source)
            if label == 'Item 14' and source[start:end].endswith('partiv'):
                end -= len('partiv')
            missing_spans.append({'source_item': label, 'normalized_start': start, 'normalized_end': end,
                                  'characters_including_section_heading': end-start})
        metadata = json.loads((source_dir / (key + '.meta.json')).read_text())
        cases.append({'filing_key': key, 'source_url': metadata['original_sec_url'],
                      'corpus_chunks': len(rows), 'original_body_item_labels': sorted(body_labels),
                      'confirmed_omitted_body_items': omitted,
                      'confirmed_item_body_omission_fraction': ratio(len(omitted), len(body_labels)),
                      'confirmed_omitted_section_span_fraction': ratio(sum(s['characters_including_section_heading'] for s in missing_spans), len(source)),
                      'spans': missing_spans,
                      'note': 'Reference-only sections; their headings can survive in the TOC. This counts omitted source section spans, not unique information loss or deleted chunks.'})
    comparisons = json.loads((ROOT / 'result/finrate_source_audit_20261007/filing_comparisons.json').read_text())
    compared = [r for r in comparisons if r['status'] == 'compared']
    result = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'dataset_revision': inventory['revision'],
        'corpus': {'companies': len(companies), 'title_filing_groups': inventory['filing_groups_by_ticker_date_form'],
                   'chunks': len(documents), 'title_date_min': inventory['min_title_date'], 'title_date_max': inventory['max_title_date'],
                   'company_year_presence': ratio(company_years, len(companies)*6),
                   'companies_with_at_least_one_record_in_every_year': ratio(inventory['companies_present_in_all_six_calendar_years'], len(companies)),
                   'year_counts': inventory['year_counts'], 'form_label_counts': inventory['form_counts']},
        'qa': {'task_coverage': task_coverage, 'all_gold_referenced_chunks': ratio(len(all_refs), len(documents)),
               'chunks_not_referenced_by_any_gold_qa_but_still_in_corpus': ratio(len(documents)-len(all_refs), len(documents)),
               'all_gold_referenced_filing_groups': ratio(len({doc_to_filing[x] for x in all_refs}), inventory['filing_groups_by_ticker_date_form'])},
        'source_audit': {'selected': len(comparisons), 'statuses': dict(Counter(r['status'] for r in comparisons)),
                         'compared_filing_coverage': ratio(len(compared), inventory['filing_groups_by_ticker_date_form']),
                         'companies_compared': len({r['ticker'] for r in compared}),
                         'confirmed_omission_cases': cases,
                         'confirmed_documents_with_omissions_among_compared_lower_bound': ratio(len(cases), len(compared)),
                         'global_deleted_chunk_fraction': None,
                         'global_deleted_chunk_fraction_status': 'not_identifiable_from_published_artifacts',
                         'why': 'No pre-filter chunk inventory/count and no original chunk-generation code are published. Source Items and released chunks are not one-to-one. Only a non-random subset has original-text comparison.'},
        'interpretation_limits': [
            'A company-year presence is at least one filing, not complete annual/quarterly history or SEC filing recall.',
            'Dates and form labels in corpus titles differ from SEC metadata in some cases; 10-K includes 10-K/A.',
            'The two confirmed omission cases were diagnostic findings; their rates do not estimate the corpus-wide rate.',
            'Normalized character fractions include section headings and exclude whitespace/punctuation; not token, chunk, or unique-fact loss.',
            'Exact 100-character window mismatches from the prior audit are not a deletion rate.',
            'Gold QA non-reference is not corpus exclusion.',
            'Primary text comparison excludes linked exhibits, images, layout and exact numeric-sign fidelity.',
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'statistics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    with (OUT / 'confirmed_omissions.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filing_key', 'source_items', 'confirmed_omitted_body_items', 'item_percent', 'confirmed_span_characters', 'source_characters', 'span_percent', 'source_url'])
        writer.writeheader()
        for case in cases:
            items = case['confirmed_item_body_omission_fraction']
            spans = case['confirmed_omitted_section_span_fraction']
            writer.writerow({'filing_key': case['filing_key'], 'source_items': items['denominator'], 'confirmed_omitted_body_items': items['numerator'], 'item_percent': items['percent'], 'confirmed_span_characters': spans['numerator'], 'source_characters': spans['denominator'], 'span_percent': spans['percent'], 'source_url': case['source_url']})
    print(json.dumps({'company_year_presence': result['corpus']['company_year_presence'],
                      'all_year_companies': result['corpus']['companies_with_at_least_one_record_in_every_year'],
                      'qa': result['qa'], 'audit_coverage': result['source_audit']['compared_filing_coverage'],
                      'cases': [{k: c[k] for k in ['filing_key', 'confirmed_item_body_omission_fraction', 'confirmed_omitted_section_span_fraction']} for c in cases]}, indent=2))


if __name__ == '__main__':
    main()
