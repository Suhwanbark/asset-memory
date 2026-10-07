from pathlib import Path
import tempfile
import unittest

from finrate_v5_development import copy_frozen_tree, verify_frozen
from finrate_poc.runtime import file_digest


class FrozenDevelopmentTests(unittest.TestCase):
    def test_reuse_preserves_failed_outputs_and_refuses_changed_copies(self):
        with tempfile.TemporaryDirectory() as tmp:
            origin = Path(tmp) / 'old'; target = Path(tmp) / 'new'; origin.mkdir()
            (origin / 'prediction.json').write_text('{"status":"failed"}\n')
            hashes = copy_frozen_tree(origin, target)
            self.assertEqual(hashes, copy_frozen_tree(origin, target))
            (target / 'prediction.json').write_text('{"status":"completed"}\n')
            with self.assertRaises(ValueError): copy_frozen_tree(origin, target)

    def test_code_and_input_mutation_are_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); code = root / 'method.py'; data = root / 'cohort.json'
            code.write_text('old code'); data.write_text('frozen questions')
            manifest = {'code_sha256': {code.name: file_digest(code)},
                        'input_sha256': {data.name: file_digest(data)}}
            verify_frozen(root, manifest)
            code.write_text('new code')
            with self.assertRaises(ValueError): verify_frozen(root, manifest)
            code.write_text('old code'); data.write_text('replacement questions')
            with self.assertRaises(ValueError): verify_frozen(root, manifest)


if __name__ == '__main__': unittest.main()
