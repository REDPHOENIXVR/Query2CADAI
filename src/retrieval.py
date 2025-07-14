import os
import json
import threading
import numpy as np

from src.logger import get_logger
from src.learning_db import log_feedback

logger = get_logger("retrieval")

class Retriever:
    """
    Persistent Retriever with disk-based FAISS index and examples manifest.
    Thread-safe with RLock. Supports positive and negative example addition.

    Methods
    -------
    add_example(query, code, embedding=None, **kwargs)
        Add a positive example. If embedding is None and status=="good", compute embedding automatically.
    add_negative_example(query, error_msg)
        Add a negative example (not indexed).
    search(embedding, k=5)
        Search top-k similar examples in the index.
    ensure_example(query, code, status="good", **kwargs)
        Public helper: add example, always computes embedding if needed.
    """

    def __init__(self, data_dir="data/examples", embedding_dim=768):
        self.data_dir = data_dir
        self.index_path = os.path.join(data_dir, "faiss.index")
        self.examples_path = os.path.join(data_dir, "examples.jsonl")
        self.lock = threading.RLock()
        self.embedding_dim = embedding_dim
        self.index = None
        self.examples = []
        self._st_model = None  # cache for sentence-transformers model
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

    def _build_fresh(self, factory=None):
        """
        Build a fresh FAISS index and clear examples.

        Parameters
        ----------
        factory : str, optional
            Optional index factory string for faiss (default: None for IndexFlatL2).
        """
        import faiss
        self.examples = []
        if factory is not None:
            self.index = faiss.index_factory(self.embedding_dim, factory)
        else:
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

    def _get_embedding(self, text):
        """
        Compute embedding for the given text using sentence-transformers.
        Lazily loads model and caches it. Logs and returns None if unavailable.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.warning("[retrieval/embedding] sentence-transformers not installed; cannot compute embedding.")
            return None
        if self._st_model is None:
            try:
                self._st_model = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info("[retrieval/embedding] Loaded sentence-transformers model: all-MiniLM-L6-v2")
            except Exception as e:
                logger.warning(f"[retrieval/embedding] Could not load model: {e}")
                return None
        emb = self._st_model.encode([text])[0]
        logger.info("[retrieval/embedding] Embedding generated and cached.")
        return emb

    def add_example(self, query, code, embedding=None, **kwargs):
        """
        Add a positive example (stored in index and manifest).
        Extra fields: tags (list), notes (str), status (default: good).
        If embedding is None and status=='good', compute embedding automatically.
        If embedding cannot be computed, log warning and still add to manifest.

        Handles dynamic FAISS index dimension: If current index is None or has mismatched dimension,
        rebuilds index to match embedding dimension.
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
                if embedding is None:
                    embedding = self._get_embedding(query)
                    if embedding is None:
                        logger.warning("[retrieval/embedding] Example NOT added to FAISS (no embedding).")
                        self._save()
                        logger.info(f"Example added: status={example['status']} (manifest only)")
                        log_feedback('macro', query, code, True)
                        return
                emb_dim = len(embedding)
                # If index is not built or dimension mismatches, rebuild it with correct dimension
                if self.index is None or getattr(self.index, 'd', 0) != emb_dim:
                    logger.info(f"[retrieval] Rebuilding FAISS index with dimension {emb_dim} to match embedding.")
                    self.embedding_dim = emb_dim
                    self._build_fresh()
                self.index.add(np.array([embedding]).astype(np.float32))
                logger.info(f"[retrieval/embedding] Example embedded and added to FAISS.")
            self._save()
            logger.info(f"Example added: status={example['status']}")
            # Feedback logging
            log_feedback('macro', query, code, True)

    def ensure_example(self, query, code, status="good", **kwargs):
        """
        Public helper: add example, always computes embedding if needed.
        """
        self.add_example(query, code, embedding=None, status=status, **kwargs)

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
            # Feedback logging
            log_feedback('macro', query, error_msg, False)

    def search(self, embedding, k=5):
        """
        Search top-k similar examples in the index.

        Parameters
        ----------
        embedding : np.ndarray
            The embedding to search for.
        k : int
            Number of results to return.

        Returns
        -------
        list
            List of matching example dicts (possibly empty).
        """
        with self.lock:
            if self.index is None or getattr(self.index, "ntotal", 0) == 0:
                logger.warning("[retrieval/search] FAISS index empty or unavailable, returning empty list.")
                return []
            D, I = self.index.search(np.array([embedding]).astype(np.float32), k)
            return [self.examples[i] for i in I[0] if i < len(self.examples)]

# Note: self.lock (RLock) is used for both reading and writing for simplicity.
# Could use context managers or separate read/write locks for more complex needs,
# but RLock is safe for current usage.