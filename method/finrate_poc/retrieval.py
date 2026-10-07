"""Shared CPU embedding passages and reproducible embedding/hybrid baselines."""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import re

from .runtime import check_disk, cpu_only, digest, write_json

EMBEDDING_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
EMBEDDING_PATH = Path("/home/hob/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots") / EMBEDDING_REVISION


def terms(text):
    return re.findall(r"[a-z0-9]+", text.casefold())


class CpuEmbedding:
    def __init__(self, path=EMBEDDING_PATH):
        cpu_only()
        check_disk()
        import torch
        from transformers import AutoModel, AutoTokenizer
        torch.set_num_threads(2)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModel.from_pretrained(path, local_files_only=True).to("cpu").eval()
        assert next(self.model.parameters()).device.type == "cpu"

    def passages(self, docs):
        """Exact source spans, 192 content tokens plus a shared title prefix."""
        output = []
        for doc in docs:
            text = doc["text"]
            offsets = self.tokenizer(text, add_special_tokens=False,
                                     return_offsets_mapping=True, verbose=False)["offset_mapping"]
            for begin in range(0, len(offsets), 160):
                window = offsets[begin:begin+192]
                if not window:
                    continue
                start, end = window[0][0], window[-1][1]
                output.append({"id": f"{doc['_id']}:{start}:{end}", "source_id": doc["_id"],
                               "title": doc["title"], "start": start, "end": end,
                               "text": text[start:end]})
        return output

    def encode(self, texts):
        import numpy as np
        vectors = []
        for begin in range(0, len(texts), 16):
            check_disk()
            encoded = self.tokenizer(texts[begin:begin+16], padding=True, truncation=True,
                                     max_length=256, return_tensors="pt")
            assert all(v.device.type == "cpu" for v in encoded.values())
            with self.torch.inference_mode():
                states = self.model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1)
                pooled = (states * mask).sum(1) / mask.sum(1).clamp(min=1)
                vectors.append(self.torch.nn.functional.normalize(pooled, p=2, dim=1).numpy())
        return np.concatenate(vectors, axis=0) if vectors else np.empty((0, 384), dtype="float32")


class Index:
    def __init__(self, passages, matrix, embedder):
        import numpy as np
        self.passages, self.matrix, self.embedder = passages, np.asarray(matrix), embedder
        if len(passages) != len(matrix) or len({p['id'] for p in passages}) != len(passages):
            raise ValueError("Index shape or ID mismatch")
        self.tf = [Counter(terms(p["title"] + " " + p["text"])) for p in passages]
        self.lengths = np.asarray([sum(row.values()) for row in self.tf], dtype=float)
        self.avg_length = float(self.lengths.mean()) if len(self.lengths) else 1.0
        df = Counter(t for row in self.tf for t in row)
        self.idf = {t: math.log(1 + (len(passages) - n + .5) / (n + .5)) for t, n in df.items()}

    def bm25(self, question):
        import numpy as np
        scores = np.zeros(len(self.passages))
        for term in set(terms(question)):
            frequency = np.asarray([row[term] for row in self.tf], dtype=float)
            norm = 1.2 * (.25 + .75 * self.lengths / max(self.avg_length, 1))
            scores += self.idf.get(term, 0) * frequency * 2.2 / (frequency + norm + 1e-12)
        return scores

    def search(self, question, *, method="embedding", k=20):
        import numpy as np
        if method not in {"embedding", "hybrid"}:
            raise ValueError(method)
        dense = self.matrix @ self.embedder.encode([question])[0]
        dense_order = np.argsort(-dense, kind="stable")
        scores = dense
        if method == "hybrid":
            sparse_order = np.argsort(-self.bm25(question), kind="stable")
            scores = np.zeros(len(self.passages))
            for order in (dense_order, sparse_order):
                scores[order] += .5 / (60 + np.arange(len(order)) + 1)
        order = np.argsort(-scores, kind="stable")[:k]
        return [{**self.passages[i], "score": float(scores[i])} for i in order]

    @classmethod
    def build(cls, docs, folder):
        import numpy as np
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        embedder = CpuEmbedding()
        passages = embedder.passages(docs)
        packet_hash = digest(json.dumps(passages, sort_keys=True))
        manifest_path = folder / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest["passages_sha256"] != packet_hash:
                raise ValueError("Frozen index inputs changed")
            matrix = np.load(folder / "vectors.npy", allow_pickle=False)
        else:
            # Explicit metadata is equally available to dense and sparse arms.
            matrix = embedder.encode([p["title"][:150] + "\n" + p["text"] for p in passages])
            check_disk()
            np.save(folder / "vectors.npy", matrix, allow_pickle=False)
            write_json(folder / "passages.json", passages, immutable=True)
            write_json(manifest_path, {"passages_sha256": packet_hash,
                       "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                       "embedding_revision": EMBEDDING_REVISION, "device": "cpu",
                       "hybrid": "equal-weight reciprocal-rank fusion, k=60",
                       "not_exact_finrate_reproduction": True}, immutable=True)
        return cls(passages, matrix, embedder)
