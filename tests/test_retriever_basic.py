import numpy as np
from src.retrieval import Retriever

def test_retriever_basic(tmp_path):
    # Ensure the parts index builds before retrieval test
    try:
        from src.build_parts_index import PartIndex
        PartIndex().build_index()
    except Exception:
        pass
    d = tmp_path / "examples"
    d.mkdir()
    r = Retriever(str(d))
    emb = np.ones(768, dtype=np.float32)
    r.add_example("circle", "macro code", emb)
    results = r.search(emb, k=1)
    assert len(results) == 1
    assert results[0]["query"] == "circle"
    r.add_negative_example("fail_query", "fail msg")
    assert r.examples[-1]["status"] == "bad"