from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import os
from finrate_asset_prefetch import publish_completed


class PrefetchTests(unittest.TestCase):
    def test_complete_asset_published_and_original_bytes_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'staging'; source.mkdir()
            (source / 'completed.json').write_text('{"done":true}')
            (source / 'bank.sqlite').write_bytes(b'fixed snapshot bytes')
            target = Path(tmp) / 'published/A'
            self.assertTrue(publish_completed(source, target))
            self.assertEqual((target / 'bank.sqlite').read_bytes(), b'fixed snapshot bytes')
            self.assertFalse(source.exists())

    def test_does_not_overwrite_consumer_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'staging'; source.mkdir()
            (source / 'completed.json').write_text('{}')
            target = Path(tmp) / 'consumer'; target.mkdir()
            (target / 'build_manifest.json').write_text('consumer has begun')
            self.assertFalse(publish_completed(source, target))
            self.assertEqual((target / 'build_manifest.json').read_text(), 'consumer has begun')
            self.assertTrue(source.exists())

    def test_incomplete_asset_cannot_be_published(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'staging'; source.mkdir()
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                publish_completed(source, Path(tmp) / 'target')

    def test_concurrent_consumer_start_wins_without_losing_either_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'staging'; source.mkdir()
            (source / 'completed.json').write_text('{}')
            target = Path(tmp) / 'consumer'
            rename = os.rename
            def consumer_starts_then_rename(src, dst):
                target.mkdir()
                (target / 'build_manifest.json').write_text('concurrent consumer')
                return rename(src, dst)
            with patch('finrate_asset_prefetch.os.rename', consumer_starts_then_rename):
                self.assertFalse(publish_completed(source, target))
            self.assertEqual((target / 'build_manifest.json').read_text(), 'concurrent consumer')
            self.assertTrue((source / 'completed.json').exists())


if __name__ == '__main__': unittest.main()
