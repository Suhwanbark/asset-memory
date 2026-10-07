"""Release an explicitly reviewed development method for one heldout run."""
import argparse
import json
from pathlib import Path

from finrate_poc.runtime import ROOT,file_digest,write_json
from .config import BASE,POLICY
from .evaluate import prediction_grid,verify_online


def freeze(review_path):
    verify_online();prediction_grid('dev')
    review_path=Path(review_path);review=json.loads(review_path.read_text())
    required={'decision':'freeze_for_one_heldout_run',
      'development_report_sha256':file_digest(BASE/'dev/report.json'),
      'online_manifest_sha256':file_digest(BASE/'online_manifest.json'),
      'all_questions_and_failures_retained':True,'heldout_results_used':False}
    if any(review.get(k)!=v for k,v in required.items()) or not review.get('rationale') or not review.get('limitations'):
        raise ValueError('A concrete review of these development results is required')
    files=[BASE/'online_manifest.json',BASE/'protocol.json',BASE/'cohort/manifest.json',
      BASE/'cohort/evaluation_inputs.json',BASE/'cohort/gold/evaluation.json',BASE/'dev/report.json',BASE/'dev/source_grounding.json',BASE/'dev/diagnostics.json',
      BASE/'index/manifest.json',BASE/'corpus.sqlite',BASE/'reader_banks/manifest.json',
      BASE/'reader_banks/bank.sqlite',BASE/'reader_banks/flat.sqlite',BASE/'index/chunks.sqlite',
      BASE/'index/vectors.npy',BASE/'index/ids.json']
    # Hash bytes; do not semantically read the heldout questions or gold here.
    release={'policy':POLICY,'online_manifest_sha256':file_digest(BASE/'online_manifest.json'),
      'review_sha256':file_digest(review_path),'input_sha256':{str(p.relative_to(BASE)):file_digest(p) for p in files},
      'heldout_release_count':1,'repeat_until_favorable_forbidden':True}
    write_json(BASE/'release.json',release,immutable=True);return release


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('review');a=p.parse_args();freeze(a.review)
