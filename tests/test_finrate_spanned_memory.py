import json
from pathlib import Path
import tempfile
import unittest

from finrate_spanned_memory import build_asset, materialize_fact, segment_batch
from finrate_memory_audit import audit
from finrate_poc.runtime import file_digest


class SpanModelFixture:
    def __init__(self): self.calls = 0
    def invoke(self, prompt, folder, **kwargs):
        self.calls += 1
        p = json.loads(prompt.rsplit('\n', 1)[1])
        if 'windows' in p:
            return json.dumps({'facts': [{'subject': p['asset'], 'topic': 'manufacturer',
                'predicate': 'arrangement', 'value': w['spans'][0]['text'], 'period': None,
                'modality': 'reported', 'source_id': w['source_id'],
                'span_start': w['spans'][0]['id'], 'span_end': w['spans'][-1]['id']}
                for w in p['windows'] if w['spans']]})
        return json.dumps({'links': [{'old_id': p['past'][0]['id'], 'new_id': p['new'][0]['id'],
            'relation': 'updates', 'rationale': 'Source reports an additional manufacturer.'}]})


class SpannedMemoryTests(unittest.TestCase):
    def test_original_characters_and_offsets_are_preserved(self):
        source = 'A table | x\n |  10\t units\n' * 120
        payload, lookup = segment_batch({'asset': 'A', 'published': '2025-02-01', 'windows': [
            {'source_id': 's', 'title': 's', 'start': 45, 'text': source}]})
        self.assertEqual(''.join(s['text'] for s in payload['windows'][0]['spans']), source)
        for span in lookup.values():
            self.assertEqual(span['text'], source[span['start']-45:span['end']-45])

    def test_cross_source_reversed_unknown_and_oversized_ranges_are_rejected(self):
        _, lookup = segment_batch({'asset': 'A', 'published': '2025-02-01', 'windows': [
            {'source_id': 'a', 'title': 'a', 'start': 0, 'text': 'a' * 4500},
            {'source_id': 'b', 'title': 'b', 'start': 0, 'text': 'different source text'}]})
        base = {'source_id': 'a', 'span_start': 'W0000S0000', 'span_end': 'W0000S0000'}
        for changes in [{'span_end': 'W0001S0000'}, {'source_id': 'b'},
                        {'span_start': 'W0000S0001'}, {'span_end': 'unknown'},
                        {'span_end': 'W0000S0005'}]:
            with self.assertRaises((ValueError, KeyError)): materialize_fact({**base, **changes}, lookup)

    def test_real_bank_update_uses_spans_and_preserves_parent(self):
        docs = [{'_id': 'a', 'title': 'A_2025-02-01_10-K', 'text': 'We use one manufacturer.'},
                {'_id': 'b', 'title': 'A_2025-02-01_10-K', 'text': 'This manufacturer supplies all components.'}]
        delta = [{'_id': 'c', 'title': 'A_2025-05-01_10-Q', 'text': 'We added a second manufacturer.'}]
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp) / 'q1'; child = Path(tmp) / 'q2'; model = SpanModelFixture()
            build_asset(docs, parent, model); parent_hash = file_digest(parent / 'bank.sqlite')
            report = build_asset(delta, child, model, parent=parent)
            self.assertEqual(report['total_sources'], 3)
            self.assertEqual(file_digest(parent / 'bank.sqlite'), parent_hash)
            checked = audit(child / 'bank.sqlite')
            self.assertEqual(checked['quote_alignment'], {'exact': 3})
            self.assertEqual(checked['links'], 1)
            before = model.calls; build_asset(delta, child, model, parent=parent)
            self.assertEqual(model.calls, before)


if __name__ == '__main__': unittest.main()
