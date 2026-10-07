"""Model-free checks of sandbox isolation, contamination rules and the per-question trace."""
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sandbox import contamination
from sandbox.environment import DockerConfig, DockerEnvironment
from sandbox.run import load_questions, run_batch, run_query
from sandbox.session import CodexSession

QUESTION = ("How did Houston American Energy revise its description of the risk that drilling services in "
            "the Permian Basin may be costly or unavailable between 2021 and 2022?")
THREAD = "0199a0e0-0000-7000-8000-000000000001"
FEATURES = b"shell_tool stable true\nunified_exec stable true\ngoals stable true\n" \
           b"workspace_dependencies stable true\nmemories stable true\napps stable true\n"


def event(item):
    return {"type": "item.completed", "item": item}


def search(*queries):
    return event({"type": "web_search", "query": queries[0], "action": {"type": "search", "queries": list(queries)}})


def shell(command):
    return event({"type": "command_execution", "command": command, "aggregated_output": "", "exit_code": 0})


def page(action, url, title):
    """Codex 0.159 web_search item: action "other" = opened page, "search" = hit list."""
    return event({"type": "web_search", "query": "", "action": {"type": action},
                  "results": [{"type": "text_result", "domain": url.split("/")[2], "title": title, "url": url}]})


CLEAN_EVENTS = [
    {"type": "thread.started", "thread_id": THREAD},
    search("Houston American Energy 10-K 2022 Permian drilling services"),
    page("other", "https://www.sec.gov/a.htm", "Houston American Energy 10-K"),
    shell("bash -lc 'curl -A \"Research Agent research@example.com\" https://www.sec.gov/cgi-bin/browse-edgar'"),
    event({"type": "agent_message", "text": "In 2022 the company added demand language."}),
    {"type": "turn.completed", "usage": {"input_tokens": 1000, "cached_input_tokens": 200,
                                         "output_tokens": 300, "reasoning_output_tokens": 100}},
]


def tar_bytes(files):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def rollout(*urls):
    lines = [{"type": "response_item", "payload": {"type": "web_search_call", "status": "completed",
                                                   "action": {"type": "open_page", "url": url}}} for url in urls]
    return {"sessions/2026/10/07/rollout-1.jsonl": "\n".join(json.dumps(line) for line in lines).encode()}


def scan(events=(), work=None, sessions=None):
    with tempfile.TemporaryDirectory() as d:
        directory = Path(d)
        (directory / "turn_01.jsonl").write_text("\n".join(json.dumps(e) for e in events))
        if work is not None:
            (directory / "work.tar").write_bytes(tar_bytes(work))
        if sessions is not None:
            (directory / "sessions.tar").write_bytes(tar_bytes(sessions))
        return contamination.detect(directory, QUESTION)


class ContaminationTests(unittest.TestCase):
    def test_ordinary_filing_research_is_clean(self):
        self.assertEqual(scan(CLEAN_EVENTS)["status"], "clean")

    def test_benchmark_name_in_any_search_query_blocks(self):
        self.assertEqual(scan([search("Fin-RATE dataset answers")])["status"], "blocked")
        # Codex lists extra queries only under action.queries.
        self.assertEqual(scan([search("Houston American 10-K", "GGLabYale Fin RATE")])["status"], "blocked")

    def test_dataset_download_command_blocks(self):
        report = scan([shell("curl -L https://huggingface.co/datasets/GGLabYale/Fin-RATE/resolve/main/qa.json")])
        self.assertEqual(report["status"], "blocked")

    def test_opened_dataset_page_blocks_but_search_hit_is_suspect(self):
        viewer = "https://huggingface.co/datasets/GGLabYale/Fin-RATE/viewer"
        self.assertEqual(scan([page("other", viewer, "GGLabYale/Fin-RATE at main")])["status"], "blocked")
        self.assertEqual(scan([page("search", viewer, "GGLabYale/Fin-RATE at main")])["status"], "suspect")

    def test_opened_dataset_page_in_session_rollout_blocks(self):
        report = scan(sessions=rollout("https://www.sec.gov/Archives/edgar/data/1156041/a.htm",
                                       "https://huggingface.co/datasets/GGLabYale/Fin-RATE/viewer"))
        self.assertEqual(report["status"], "blocked")
        self.assertIn("sessions.tar", report["scanned"])

    def test_declared_paper_source_blocks(self):
        answer = {"answer": "x", "sources": ["https://arxiv.org/abs/2602.07294"]}
        self.assertEqual(scan(work={"./answer.json": json.dumps(answer).encode()})["status"], "blocked")

    def test_advisory_signals_are_suspect_not_blocked(self):
        self.assertEqual(scan([shell("curl https://github.com/someone/edgar-tools")])["status"], "suspect")
        self.assertEqual(scan([event({"type": "agent_message", "text": "Per FinRATE..."})])["status"], "suspect")
        copied = " ".join(QUESTION.split()[:14])
        report = scan([search(copied)])
        self.assertEqual(report["status"], "suspect")
        self.assertIn("verbatim_query_search:", report["advisory_summary"])

    def test_generic_finrate_field_names_are_not_markers(self):
        self.assertEqual(scan([shell("python -c 'key_points = []; answer = 1'")])["status"], "clean")


class IsolationTests(unittest.TestCase):
    def test_container_is_locked_down(self):
        argv = DockerEnvironment(DockerConfig(image="sha256:test")).create_command()
        for flag in ("--read-only", "no-new-privileges", "ALL", "--pids-limit"):
            self.assertIn(flag, argv)
        self.assertNotIn("-v", argv)
        self.assertFalse(any("type=bind" in arg or "docker.sock" in arg for arg in argv))
        for host in contamination.BLOCKED_HOSTS:
            self.assertIn(host + ":0.0.0.0", argv)

    def test_codex_argv_has_web_search_and_keeps_rollout(self):
        session = CodexSession(DockerEnvironment(DockerConfig(image="sha256:test")), Path("."), model="gpt-6-luna")
        session.features = {"memories", "apps", "shell_tool"}
        argv = session.argv()
        self.assertIn("--search", argv)
        self.assertNotIn("--ephemeral", argv)
        pairs = list(zip(argv, argv[1:]))
        for feature in ("memories", "apps"):
            self.assertIn(("--disable", feature), pairs)
        self.assertEqual(argv[argv.index("--model") + 1], "gpt-6-luna")


class FakeEnvironment:
    """In-memory stand-in for DockerEnvironment that replays a Codex turn."""
    created = []

    def __init__(self, config, *, events=CLEAN_EVENTS, answer=None, sessions=None, timeout=False):
        self.config, self.name, self.container_id, self.stopped = config, "fake", None, False
        self.events, self.answer, self.sessions, self.timeout = events, answer, sessions or {}, timeout
        self.files, self.prompts, self.frozen = {}, [], False
        FakeEnvironment.created.append(self)

    async def start(self):
        self.container_id = "cid"

    async def put(self, path, data):
        self.files[path] = data

    async def exec(self, argv, *, data=None, timeout=30, output=None, limit=0):
        if argv[:2] == ["codex", "--version"]:
            return 0, b"codex-cli 0.159.2\n", b""
        if argv[:3] == ["codex", "features", "list"]:
            return 0, FEATURES, b""
        self.prompts.append(data.decode())
        output.write_text("\n".join(json.dumps(e) for e in self.events[:2 if self.timeout else None]))
        if self.answer is not None:
            self.files["/work/answer.json"] = json.dumps(self.answer).encode()
        if self.timeout:
            raise TimeoutError()
        return 0, b"", b""

    async def read_file(self, path):
        return self.files.get(path)

    async def collect(self, destination):
        work = {"." + path[len("/work"):]: data for path, data in self.files.items() if path.startswith("/work/")}
        (destination / "work.tar").write_bytes(tar_bytes(work))
        (destination / "sessions.tar").write_bytes(tar_bytes(self.sessions))
        return []

    async def freeze(self):
        self.frozen = True

    async def close(self):
        self.stopped = True


def factory(**kwargs):
    return lambda config: FakeEnvironment(config, **kwargs)


QUESTION_ROW = {"q_id": "lt_001", "task": "LT-QA", "question": QUESTION}
COMMON = dict(docker=DockerConfig(image="sha256:test"), auth=b'{"tokens":{"access_token":"x"}}',
              model="gpt-6-luna", effort="medium", timeout=900, template="Q:\n{question}\n")


class RunTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        FakeEnvironment.created.clear()
        self.root = Path(tempfile.mkdtemp())

    def test_inputs_keep_only_question_fields(self):
        path = self.root / "inputs.json"
        path.write_text(json.dumps([{**QUESTION_ROW, "gold_doc_ids": ["doc_1"]}]))
        self.assertEqual(load_questions(path), [QUESTION_ROW])
        path.write_text(json.dumps([{**QUESTION_ROW, "answer": "gold", "key_points": []}]))
        with self.assertRaises(ValueError):
            load_questions(path)

    async def test_completed_question_keeps_trace_and_hides_ids(self):
        answer = {"answer": "It added demand language.", "sources": ["https://www.sec.gov/a.htm"]}
        directory = self.root / "lt_001"
        result = await run_query(QUESTION_ROW, directory, **COMMON,
                                 environment_factory=factory(answer=answer, sessions=rollout()))
        environment = FakeEnvironment.created[0]
        self.assertEqual((result["status"], result["answer_source"]), ("completed", "answer.json"))
        self.assertEqual(result["sources"], answer["sources"])
        self.assertTrue(result["score_eligible"])
        self.assertEqual(result["contamination_status"], "clean")
        self.assertEqual(result["usage"]["input_tokens"], 1000)
        self.assertEqual(result["activity"], {"web_searches": 1, "web_page_actions": 1, "commands": 1,
                                              "opened_urls": ["https://www.sec.gov/a.htm"]})
        self.assertNotIn("lt_001", environment.prompts[0])
        self.assertNotIn("LT-QA", environment.prompts[0])
        self.assertEqual(environment.files["/home/agent/.codex/auth.json"], COMMON["auth"])
        self.assertTrue(environment.frozen and environment.stopped)
        for name in ("input.json", "prompt_01.txt", "turn_01.jsonl", "session.json", "work.tar",
                     "sessions.tar", "contamination.json", "result.json", "prediction.json"):
            self.assertTrue((directory / name).exists(), name)
        prediction = json.loads((directory / "prediction.json").read_text())
        self.assertEqual((prediction["state"], prediction["answer"]), ("completed", answer["answer"]))

    async def test_final_message_is_the_fallback_answer(self):
        result = await run_query(QUESTION_ROW, self.root / "q", **COMMON, environment_factory=factory())
        self.assertEqual((result["status"], result["answer_source"]), ("completed", "final_message"))

    async def test_timeout_still_collects_and_closes(self):
        result = await run_query(QUESTION_ROW, self.root / "q", **COMMON, environment_factory=factory(timeout=True))
        environment = FakeEnvironment.created[0]
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["score_eligible"])
        self.assertTrue(environment.frozen and environment.stopped)
        self.assertTrue((self.root / "q" / "turn_01.jsonl").exists())

    async def test_contaminated_answer_is_not_score_eligible(self):
        events = [CLEAN_EVENTS[0], search("Fin-RATE LT-QA answers"), *CLEAN_EVENTS[1:]]
        result = await run_query(QUESTION_ROW, self.root / "q", **COMMON, environment_factory=factory(events=events))
        self.assertEqual((result["status"], result["contamination_status"]), ("completed", "blocked"))
        self.assertFalse(result["score_eligible"])

    async def test_batch_resumes_and_refuses_changed_settings(self):
        questions = [QUESTION_ROW, {"q_id": "ec_002", "task": "EC-QA", "question": "Compare A and B."}]
        output = self.root / "run"
        summary = await run_batch(questions, output, concurrency=2, environment_factory=factory(), **COMMON)
        self.assertEqual((summary["processed"], summary["status"]), (2, {"completed": 2}))
        self.assertEqual(len(FakeEnvironment.created), 2)
        self.assertEqual(len((output / "predictions.jsonl").read_text().splitlines()), 2)
        (output / "ec_002" / "result.json").unlink()
        await run_batch(questions, output, concurrency=2, environment_factory=factory(), **COMMON)
        self.assertEqual(len(FakeEnvironment.created), 3)
        self.assertEqual(len(list((output / "_aborted").iterdir())), 1)
        with self.assertRaises(ValueError):
            await run_batch(questions, output, concurrency=2, environment_factory=factory(),
                            **{**COMMON, "model": "gpt-6-astra"})


if __name__ == "__main__":
    unittest.main()
