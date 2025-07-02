import os
import json
import uuid
import threading

class ExampleRetriever:
    def __init__(self, data_dir, embedding_model_name="all-MiniLM-L6-v2"):
        self.data_dir = data_dir
        self.embedding_model_name = embedding_model_name
        self.examples = []
        self.ids = []
        self.embeddings = None
        self.emb_model = None
        self.index = None
        self.lock = threading.Lock()
        self._import_libraries()
        self.load_examples()
        self.build_index()

    def _import_libraries(self):
        try:
            import faiss
            import sentence_transformers
        except ImportError:
            raise ImportError(
                "Required libraries for retrieval not found. "
                "Please install 'sentence_transformers' and 'faiss-cpu'."
            )

    def load_examples(self):
        self.examples = []
        self.ids = []
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)
        for fname in os.listdir(self.data_dir):
            if fname.endswith(".json"):
                fpath = os.path.join(self.data_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if "query" in data and "code" in data:
                            self.examples.append(data)
                            self.ids.append(fname)
                except Exception as e:
                    print(f"Warning: Could not read {fname}: {e}")

    def build_index(self):
        with self.lock:
            if not self.examples:
                self.embeddings = None
                self.index = None
                return
            try:
                from sentence_transformers import SentenceTransformer
                import faiss
            except ImportError:
                raise ImportError(
                    "Required libraries for retrieval not found. "
                    "Please install 'sentence_transformers' and 'faiss-cpu'."
                )
            self.emb_model = SentenceTransformer(self.embedding_model_name)
            queries = [ex["query"] for ex in self.examples]
            self.embeddings = self.emb_model.encode(queries, show_progress_bar=False)
            dim = self.embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dim)
            self.index.add(self.embeddings)

    def get_similar_examples(self, query, k=3):
        with self.lock:
            if not self.examples or self.index is None:
                return []
            try:
                import numpy as np
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "Required libraries for retrieval not found. "
                    "Please install 'sentence_transformers' and 'faiss-cpu'."
                )
            if self.emb_model is None:
                from sentence_transformers import SentenceTransformer
                self.emb_model = SentenceTransformer(self.embedding_model_name)
            query_emb = self.emb_model.encode([query])
            D, I = self.index.search(query_emb, min(k, len(self.examples)))
            return [self.examples[i] for i in I[0]]

    def add_example(self, query, code):
        # Write to disk first
        uid = str(uuid.uuid4())
        path = os.path.join(self.data_dir, f"{uid}.json")
        data = {"query": query, "code": code}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        # Update memory
        self.examples.append(data)
        self.ids.append(f"{uid}.json")
        # Update FAISS index
        try:
            from sentence_transformers import SentenceTransformer
            import faiss
            import numpy as np
        except ImportError:
            raise ImportError(
                "Required libraries for retrieval not found. "
                "Please install 'sentence_transformers' and 'faiss-cpu'."
            )
        if self.emb_model is None:
            self.emb_model = SentenceTransformer(self.embedding_model_name)
        emb = self.emb_model.encode([query])
        if self.embeddings is None:
            self.embeddings = emb
        else:
            self.embeddings = np.vstack([self.embeddings, emb])
        if self.index is None:
            dim = emb.shape[1]
            self.index = faiss.IndexFlatL2(dim)
            self.index.add(self.embeddings)
        else:
            self.index.add(emb)