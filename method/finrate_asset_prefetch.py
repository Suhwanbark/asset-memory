"""Compute independent asset memories ahead of the serial v2 consumer.

The frozen v2 builder/model/prompts are reused verbatim in a separate staging
tree. A complete directory is published atomically only while the consumer has
not taken ownership. Existing consumer work is never replaced. No questions,
answers or scores are loaded by this worker.
"""
import argparse
import json
import os
from pathlib import Path
import time

from finrate_poc.data import documents, metadata
from finrate_poc.memory import build_asset_memory
from finrate_poc.model import Luna
from finrate_poc.runtime import ROOT, check_disk, cpu_only, file_digest, watchdog, write_json


class ConsumerOwnsAsset(RuntimeError):
    pass


def publish_completed(staging, target):
    staging, target = Path(staging), Path(target)
    if not (staging / 'completed.json').is_file():
        raise ValueError('Cannot publish an incomplete asset')
    if target.exists():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        # The consumer writes a nonempty build manifest before checking
        # completed.json. rename cannot replace that nonempty directory.
        # If only its empty directory was created concurrently, publishing the
        # complete asset still precedes its completion check and is safe.
        os.rename(staging, target)
    except (FileExistsError, OSError) as exc:
        if target.exists():
            return False
        raise exc
    return True


class PrefetchLuna(Luna):
    def __init__(self, target):
        super().__init__()
        self.target = target

    def invoke(self, *args, **kwargs):
        if self.target.exists():
            raise ConsumerOwnsAsset('Serial consumer already owns the asset; prefetch stops')
        return super().invoke(*args, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--asset', required=True)
    args = parser.parse_args()
    cpu_only(); check_disk(); os.nice(10)
    base = ROOT / 'result/finrate_luna_v1'
    manifest = json.loads((base / 'cohort/cohort_manifest.json').read_text())
    if args.asset not in manifest['assets']:
        raise ValueError('Asset is not in the frozen cohort')
    target = base / 'asset_memory/v2' / args.asset
    work = base / 'asset_prefetch_v2' / args.asset
    status = work / 'status.json'
    stop = watchdog(status)
    common = {'pid': os.getpid(), 'started': time.time(), 'asset': args.asset,
              'cpu_only': True, 'disk_stop_bytes': 10_000_000_000}
    try:
        expected = json.loads((base / 'v2/dev/run_manifest.json').read_text())['code_hashes']
        observed = {p.name: file_digest(p) for p in (ROOT / 'method/finrate_poc').glob('*.py')}
        if observed != expected:
            raise ValueError('v2 package changed; prefetch is not an identical method')
        write_json(work / 'manifest.json', {
            'asset': args.asset, 'method_version': 'v2', 'prefetch_code_sha256': file_digest(__file__),
            'method_code_sha256': observed, 'target': str(target),
            'source_manifest_sha256': file_digest(ROOT / 'data/finrate_public_v1/source_manifest.json'),
            'questions_or_scores_used': False}, immutable=True)
        if target.exists():
            write_json(status, {**common, 'state': 'skipped_consumer_already_owns_asset'})
            return
        write_json(status, {**common, 'state': 'building_asset'})
        docs = [d for d in documents(ROOT / 'data/finrate_public_v1') if metadata(d)['ticker'] == args.asset]
        build_asset_memory(docs, work / 'staging', PrefetchLuna(target), version='v2')
        staged = work / 'staging/v2' / args.asset
        completion = json.loads((staged / 'completed.json').read_text())
        published = publish_completed(staged, target)
        write_json(status, {**common, 'state': 'published_for_consumer' if published else 'consumer_won_publish_race',
                            'finished': time.time(), 'completion': completion})
    except ConsumerOwnsAsset as exc:
        write_json(status, {**common, 'state': 'stopped_consumer_took_ownership', 'reason': str(exc)})
    except Exception as exc:
        write_json(status, {**common, 'state': 'prefetch_failed', 'error': type(exc).__name__ + ': ' + str(exc)})
        raise
    finally:
        stop.set()


if __name__ == '__main__': main()
