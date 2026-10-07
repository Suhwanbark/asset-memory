import json
from pathlib import Path
import tempfile
import unittest

from finrate_autonomous_v10.agent import WRITER_PROMPT
from finrate_resume_agent_v1 import replay_values


class ResumeTest(unittest.TestCase):
    def fixture(self, root):
        initial = {'z': 1, 'a': {'y': 2, 'b': 3}}
        record = {'action': {'tool': 'read', 'arguments_json': '{}'},
                  'result': {'z': 'source', 'a': 4}}
        folder = root / 'turns/000';folder.mkdir(parents=True)
        (folder/'tool_00.json').write_text(json.dumps(record, sort_keys=True))
        for step, history in [(0, []), (1, [{'observations': [record]}])]:
            path = root / 'turns' / f'{step:03d}' / 'request.json'
            path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps({'prompt': WRITER_PROMPT+'\n'+json.dumps({'initial': initial, 'history': history})}))
        return initial, record

    def test_original_order_restored_without_content_change(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);initial,record=self.fixture(root)
            sorted_initial=json.loads(json.dumps(initial,sort_keys=True))
            restored,observations=replay_values(sorted_initial,root,writer=True)
            self.assertEqual(restored,initial)
            self.assertEqual(json.dumps(restored),json.dumps(initial))
            self.assertEqual(json.dumps(next(iter(observations.values()))),json.dumps(record))

    def test_changed_observation_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);initial,_=self.fixture(root)
            (root/'turns/000/tool_00.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'content changed'):
                replay_values(initial,root,writer=True)


if __name__=='__main__':unittest.main()
