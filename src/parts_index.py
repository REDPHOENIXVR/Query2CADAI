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