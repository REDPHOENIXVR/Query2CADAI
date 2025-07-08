import os
import tempfile
import shutil
import pytest

def has_sentence_transformers():
    try:
        import sentence_transformers
        return True
    except ImportError:
        return False

@pytest.mark.skipif(not has_sentence_transformers(), reason="sentence-transformers not installed")
def test_add_example_no_embedding():
    """
    Ensures retriever.add_example works without embedding argument,
    and that the example is indexed if sentence-transformers is available.
    """
    from src.retrieval import Retriever
    import numpy as np

    # Use a temporary directory to avoid polluting real data
    tmpdir = tempfile.mkdtemp()
    try:
        retriever = Retriever(data_dir=tmpdir)
        initial_index_size = getattr(retriever.index, "ntotal", 0)
        query = "Test retrieve query"
        code = "print('Hello world')"
        retriever.add_example(query, code)
        # After adding, index should increase if embedding available
        final_index_size = getattr(retriever.index, "ntotal", 0)
        assert final_index_size == initial_index_size + 1, (
            f"Index size did not increase: {initial_index_size} -> {final_index_size}"
        )
        # Ensure manifest is updated
        assert any(ex["query"] == query for ex in retriever.examples)
    finally:
        shutil.rmtree(tmpdir)