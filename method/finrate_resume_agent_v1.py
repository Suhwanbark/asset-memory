"""Replay the frozen agent with the exact original JSON object key order.

Artifact serialization sorts keys; model prompts did not. Recover key order
from original request transcripts, while requiring identical decoded values.
No prompt, action, observation, token budget or agent code is relaxed.
"""
import json
from pathlib import Path
from types import FunctionType, SimpleNamespace

from finrate_autonomous_v10 import agent


def replay_values(initial, folder, *, writer):
    prefix = (agent.WRITER_PROMPT if writer else agent.QA_PROMPT) + '\n'
    folder = Path(folder)
    requests = sorted(folder.glob('turns/*/request.json'))
    if not requests:
        return initial, {}
    ordered_initial = None
    observations = {}
    for path in requests:
        prompt = json.loads(path.read_text())['prompt']
        if not prompt.startswith(prefix):
            raise ValueError('Saved agent system prompt changed')
        state = json.loads(prompt[len(prefix):])
        if state['initial'] != initial:
            raise ValueError('Saved initial data changed')
        if ordered_initial is None:
            ordered_initial = state['initial']
        for step, turn in enumerate(state['history']):
            for n, record in enumerate(turn['observations']):
                if 'action' not in record:
                    continue
                saved = folder / 'turns' / f'{step:03d}' / f'tool_{n:02d}.json'
                text = saved.read_text()
                if json.loads(text) != record:
                    raise ValueError('Saved observation content changed')
                if text in observations and json.dumps(observations[text], ensure_ascii=False) != json.dumps(record, ensure_ascii=False):
                    raise ValueError('Inconsistent original observation serialization')
                observations[text] = record
    return ordered_initial, observations


def run(initial, workspace, model, folder, *, writer=False):
    initial, observations = replay_values(initial, folder, writer=writer)

    def loads(text, *args, **kwargs):
        if not args and not kwargs and isinstance(text, str) and text in observations:
            # Return a fresh object with exactly the original insertion order.
            return json.loads(json.dumps(observations[text], ensure_ascii=False))
        return json.loads(text, *args, **kwargs)

    namespace = {**agent.run.__globals__, 'json': SimpleNamespace(dumps=json.dumps, loads=loads)}
    replay = FunctionType(agent.run.__code__, namespace, agent.run.__name__,
                          agent.run.__defaults__, agent.run.__closure__)
    replay.__kwdefaults__ = agent.run.__kwdefaults__
    return replay(initial, workspace, model, folder, writer=writer)
