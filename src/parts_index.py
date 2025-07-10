import os
import glob
import json
import logging

def _lazy_import_sentence_transformer():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer
    except ImportError:
        logging.warning("sentence-transformers not installed. PartIndex will not work fully.")
        return None

def _lazy_import_faiss():
    try:
        import faiss
        return faiss
    except ImportError:
        logging.warning("faiss not installed. PartIndex will not work fully.")
        return None

def _lazy_import_yaml():
    try:
        import yaml
        return yaml
    except ImportError:
        logging.warning("pyyaml not installed. PartIndex will not work fully.")
        return None

class PartMeta:
    def __init__(self, id, category, model, mass, tags, filename):
        self.id = id
        self.category = category
        self.model = model
        self.mass = mass
        self.tags = tags
        self.filename = filename

    def to_dict(self):
        return {
            "id": self.id,
            "category": self.category,
            "model": self.model,
            "mass": self.mass,
            "tags": self.tags,
            "filename": self.filename
        }

class PartIndex:
    def __init__(self, parts_dir="library/parts"):
        self.parts_dir = parts_dir
        self.parts = []
        self.index = None
        self.manifest = {}

    def _ensure_minimal_yaml(self, path):
        import os
        import yaml
        yml_path = os.path.splitext(path)[0] + ".yml"
        if not os.path.exists(yml_path):
            stub = {
                "category": "unknown",
                "model": os.path.splitext(os.path.basename(path))[0],
                "mass": None,
                "tags": [],
                "needs_review": True
            }
            with open(yml_path, "w") as f:
                yaml.safe_dump(stub, f)

    def build_index(self, parts_dir=None):
        import os
        import glob
        parts_dir = parts_dir or self.parts_dir
        self.parts = []
        self.index = []
        self.manifest = {}
        files = glob.glob(os.path.join(parts_dir, "*"))
        for path in files:
            if path.endswith(".STEP") or path.endswith(".FCStd"):
                self._ensure_minimal_yaml(path)
                # ... (rest of build_index logic: read YAML, add to self.parts, etc.)

    def add_example(self, query_text, file_path):
        from sentence_transformers import SentenceTransformer
        import numpy as np
        # Embed query_text
        model = SentenceTransformer("all-MiniLM-L6-v2")
        emb = model.encode([query_text])[0]
        # Add to index and manifest
        self.index.append(emb)
        self.manifest[file_path] = {"query": query_text, "embedding": emb.tolist()}
    def __init__(self, index_path, faiss_index=None):
        self.index_path = index_path
        self.faiss_index = faiss_index
        self.examples = []
        # ... other fields

    def add_example(self, example):
        self.examples.append(example)
        # Immediately push embedding and add to FAISS
        if hasattr(self, "faiss_index") and self.faiss_index is not None:
            emb = self.embed_example(example)
            self.faiss_index.add(emb)
        # ... any other logic ...

    def build_index(self):
        # ... builds the index from yaml files ...
        import os, yaml
        files = [f for f in os.listdir(self.index_path) if f.endswith(('.step', '.STEP'))]
        for f in files:
            yml = f + '.yml'
            yml_path = os.path.join(self.index_path, yml)
            if not os.path.exists(yml_path):
                # Create minimal YAML metadata using filename as model
                meta = {
                    "category": "unknown",
                    "model": os.path.splitext(f)[0],
                    "mass": None,
                    "tags": []
                }
                with open(yml_path, "w") as out:
                    yaml.safe_dump(meta, out)
        # ... continue with usual index building ...
    def __init__(self):
        self.parts = []
        self.id_to_path = {}
        self.embeddings = []
        self.index = None
        self.manifest = []

    def build_index(self, library_dir="library/parts"):
        yaml = _lazy_import_yaml()
        SentenceTransformer = _lazy_import_sentence_transformer()
        faiss = _lazy_import_faiss()
        if not (yaml and SentenceTransformer and faiss):
            logging.warning("Cannot build index due to missing dependencies.")
            return

        st_model = SentenceTransformer("all-MiniLM-L6-v2")
        part_files = glob.glob(os.path.join(library_dir, "*"))
        manifest = []
        texts = []
        embeddings = []
        id_to_path = {}
        parts = []

        for pf in part_files:
            stem = os.path.splitext(os.path.basename(pf))[0]
            yml_path = pf + ".yml"
            if not os.path.isfile(yml_path):
                continue
            with open(yml_path, "r") as f:
                meta = yaml.safe_load(f)
            # Validation: skip if missing model/category or needs_review
            if not meta or not meta.get("model") or not meta.get("category") or meta.get("needs_review", True):
                logging.warning(f"Skipping {pf}: missing model/category or needs_review=true in YAML.")
                continue
            category = meta.get("category", "")
            model = meta.get("model", "")
            mass = meta.get("mass", 0)
            tags = meta.get("tags", [])
            if not isinstance(tags, list): tags = [tags]
            part = PartMeta(stem, category, model, mass, tags, pf)
            parts.append(part)
            id_to_path[stem] = pf
            manifest.append(part.to_dict())
            text = f"{category} {model} {' '.join(tags)}"
            texts.append(text)
        if not texts:
            self.parts = []
            self.id_to_path = {}
            self.embeddings = []
            self.index = None
            self.manifest = []
            return
        embs = st_model.encode(texts)
        index = faiss.IndexFlatL2(embs.shape[1])
        index.add(embs)
        self.parts = parts
        self.id_to_path = id_to_path
        self.embeddings = embs
        self.index = index
        self.manifest = manifest

        # Persist
        os.makedirs("library/index", exist_ok=True)
        faiss.write_index(index, "library/index/faiss.index")
        with open("library/index/manifest.jsonl", "w") as f:
            for meta in manifest:
                f.write(json.dumps(meta) + "\n")

    def query(self, spec_text, k=5):
        SentenceTransformer = _lazy_import_sentence_transformer()
        faiss = _lazy_import_faiss()
        if not (SentenceTransformer and faiss and self.index is not None):
            logging.warning("PartIndex dependencies not installed or index not built.")
            return []
        st_model = SentenceTransformer("all-MiniLM-L6-v2")
        emb = st_model.encode([spec_text])
        D, I = self.index.search(emb, k)
        return [self.parts[i] for i in I[0] if i < len(self.parts)]

    def get_part_path(self, part_id):
        return self.id_to_path.get(part_id)

    @staticmethod
    def load():
        faiss = _lazy_import_faiss()
        if not faiss:
            return PartIndex()
        index_path = "library/index/faiss.index"
        manifest_path = "library/index/manifest.jsonl"
        pi = PartIndex()
        if os.path.exists(index_path) and os.path.exists(manifest_path):
            pi.index = faiss.read_index(index_path)
            with open(manifest_path) as f:
                pi.manifest = [json.loads(line) for line in f]
            pi.parts = [PartMeta(**m) for m in pi.manifest]
            pi.id_to_path = {m["id"]: m["filename"] for m in pi.manifest}
        return pi