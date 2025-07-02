import os
import json
import threading
import numpy as np

from src.logger import get_logger

logger = get_logger("retrieval")

class Retriever:
    """
    Persistent Retriever with disk-based FAISS index and examples manifest.
    Thread-safe with RLock. Supports positive and negative example addition.
    """
    def __init__(self, data_dir="data/examples", embedding_dim=768):
        self.data_dir = data_dir
        self.index_path = os.path.join(data_dir, "faiss.index")
        self.examples_path = os.path.join(data_dir, "examples.jsonl")
        self.lock = threading.RLock()
        self.embedding_dim = embedding_dim
        self.index = None
        self.examples = []
        self._import_libraries()
        self._load_or_build()

    def _import_libraries(self):
        try:
            import faiss
        except ImportError:
            raise ImportError("Please install 'faiss-cpu'.")
        try:
            import numpy
        except ImportError:
            raise ImportError("Please install 'numpy'.")

    def _load_or_build(self):
        with self.lock:
            if os.path.exists(self.index_path) and os.path.exists(self.examples_path):
                try:
                    import faiss
                    self.index = faiss.read_index(self.index_path)
                    with open(self.examples_path, "r", encoding="utf-8") as f:
                        self.examples = [json.loads(line) for line in f]
                    logger.info(f"Loaded FAISS index and {len(self.examples)} examples from disk.")
                except Exception as e:
                    logger.warning(f"Failed to load index/manifest, building fresh: {e}")
                    self._build_fresh()
            else:
                self._build_fresh()

    def _build_fresh(self):
        import faiss
        self.examples = []
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self._save()

    def _save(self):
        import faiss
        with self.lock:
            faiss.write_index(self.index, self.index_path)
            with open(self.examples_path, "w", encoding="utf-8") as f:
                for ex in self.examples:
                    f.write(json.dumps(ex) + "\n")
        logger.debug("Saved FAISS index and examples manifest.")

    def add_example(self, query, code, embedding, **kwargs):
        """
        Add a positive example (stored in index and manifest).
        Extra fields: tags (list), notes (str), status (default: good).
        """
        with self.lock:
            example = {
                "query": query,
                "code": code,
                "tags": kwargs.get("tags", []),
                "notes": kwargs.get("notes", ""),
                "status": kwargs.get("status", "good")
            }
            self.examples.append(example)
            if example["status"] == "good":
                self.index.add(np.array([embedding]).astype(np.float32))
            self._save()
            logger.info(f"Example added: status={example['status']}")

    def add_negative_example(self, query, error_msg):
        """
        Add a negative example (stored in manifest only, not FAISS).
        """
        with self.lock:
            example = {
                "query": query,
                "code": "",
                "tags": [],
                "notes": error_msg,
                "status": "bad"
            }
            self.examples.append(example)
            self._save()
            logger.info("Negative example added.")

    def search(self, embedding, k=5):
        """
        Search top-k similar examples in the index.
        """
        with self.lock:
            if self.index is None or self.index.ntotal == 0:
                logger.warning("FAISS index empty, search returns no results.")
                return []
            D, I = self.index.search(np.array([embedding]).astype(np.float32), k)
            return [self.examples[i] for i in I[0] if i < len(self.examples)]

# Note: self.lock (RLock) is used for both reading and writing for simplicity.
# Could use context managers or separate read/write locks for more complex needs,
# but RLock is safe for current usage.