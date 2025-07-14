import os
import glob
import json
import logging
from typing import List, Optional, Dict, Any, TYPE_CHECKING
from dataclasses import dataclass, asdict

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

@dataclass
class PartMeta:
    id: str
    category: str
    model: str
    mass: Optional[float]
    tags: List[str]
    filename: str
    thumbnail: Optional[str] = None  # Path to PNG thumbnail if available

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class PartIndex:
    def __init__(self, parts_dir: str = "library/parts"):
        self.parts_dir: str = parts_dir
        self.parts: List[PartMeta] = []
        self.index = None  # faiss index or None
        self.id_to_path: Dict[str, str] = {}
        self.embeddings = None
        self.manifest: List[Dict[str, Any]] = []

    def _ensure_minimal_yaml(self, path: str) -> None:
        yaml = _lazy_import_yaml()
        if yaml is None:
            return
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

    def build_index(self, parts_dir: Optional[str] = None) -> None:
        """
        Build the index for the current parts directory.

        Args:
            parts_dir (Optional[str]): If provided and different from self.parts_dir, sets self.parts_dir.
        """
        if parts_dir is not None and parts_dir != self.parts_dir:
            self.parts_dir = parts_dir

        yaml = _lazy_import_yaml()
        SentenceTransformer = _lazy_import_sentence_transformer()
        faiss = _lazy_import_faiss()
        if not (yaml and SentenceTransformer and faiss):
            logging.warning("Cannot build index due to missing dependencies.")
            self.parts = []
            self.id_to_path = {}
            self.embeddings = None
            self.index = None
            self.manifest = []
            return

        st_model = SentenceTransformer("all-MiniLM-L6-v2")
        part_files = glob.glob(os.path.join(self.parts_dir, "*"))
        manifest: List[Dict[str, Any]] = []
        texts = []
        id_to_path: Dict[str, str] = {}
        parts: List[PartMeta] = []

        for pf in part_files:
            if pf.endswith(".STEP") or pf.endswith(".FCStd"):
                self._ensure_minimal_yaml(pf)
                yml_path = os.path.splitext(pf)[0] + ".yml"
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
                mass = meta.get("mass", None)
                tags = meta.get("tags", [])
                if not isinstance(tags, list):
                    tags = [tags]
                part_id = os.path.splitext(os.path.basename(pf))[0]
                # Find thumbnail path
                thumbnails_dir = "library/thumbnails"
                thumb_path = os.path.join(thumbnails_dir, f"{part_id}.png")
                if os.path.isfile(thumb_path):
                    thumbnail = thumb_path
                else:
                    thumbnail = None
                part = PartMeta(
                    id=part_id,
                    category=category,
                    model=model,
                    mass=mass,
                    tags=tags,
                    filename=pf,
                    thumbnail=thumbnail
                )
                parts.append(part)
                id_to_path[part_id] = pf
                manifest.append(part.to_dict())
                text = f"{category} {model} {' '.join(tags)}"
                texts.append(text)

        if not texts:
            self.parts = []
            self.id_to_path = {}
            self.embeddings = None
            self.index = None
            self.manifest = []
            return

        import numpy as np
        embs = st_model.encode(texts)
        index = faiss.IndexFlatL2(embs.shape[1])
        index.add(np.array(embs, dtype=np.float32))
        self.parts = parts
        self.id_to_path = id_to_path
        self.embeddings = embs
        self.index = index
        self.manifest = manifest

        # Persist index and manifest
        os.makedirs("library/index", exist_ok=True)
        faiss.write_index(index, "library/index/faiss.index")
        with open("library/index/manifest.jsonl", "w") as f:
            for meta in manifest:
                f.write(json.dumps(meta) + "\n")

    def query(self, spec_text: str, k: int = 5) -> List[PartMeta]:
        SentenceTransformer = _lazy_import_sentence_transformer()
        faiss = _lazy_import_faiss()
        if not (SentenceTransformer and faiss and self.index is not None):
            logging.warning("PartIndex dependencies not installed or index not built.")
            return []
        st_model = SentenceTransformer("all-MiniLM-L6-v2")
        emb = st_model.encode([spec_text])
        D, I = self.index.search(emb, k)
        return [self.parts[i] for i in I[0] if 0 <= i < len(self.parts)]

    def get_part_path(self, part_id: str) -> Optional[str]:
        return self.id_to_path.get(part_id)

    def add_example(self, query_text: str, part_path: str) -> None:
        SentenceTransformer = _lazy_import_sentence_transformer()
        faiss = _lazy_import_faiss()
        yaml = _lazy_import_yaml()
        if not (SentenceTransformer and faiss and yaml):
            logging.warning("Cannot add example due to missing dependencies.")
            return
        if not os.path.isfile(part_path):
            logging.warning(f"Part file {part_path} does not exist.")
            return
        st_model = SentenceTransformer("all-MiniLM-L6-v2")
        import numpy as np
        emb = st_model.encode([query_text])[0]
        # Add to manifest and index in memory
        part_id = os.path.splitext(os.path.basename(part_path))[0]
        yml_path = os.path.splitext(part_path)[0] + ".yml"
        if not os.path.isfile(yml_path):
            self._ensure_minimal_yaml(part_path)
        with open(yml_path, "r") as f:
            meta = yaml.safe_load(f)
        category = meta.get("category", "")
        model = meta.get("model", "")
        mass = meta.get("mass", None)
        tags = meta.get("tags", [])
        if not isinstance(tags, list):
            tags = [tags]
        part_meta = PartMeta(
            id=part_id,
            category=category,
            model=model,
            mass=mass,
            tags=tags,
            filename=part_path
        )
        # Only add if not already present
        if all(pm.id != part_id for pm in self.parts):
            self.parts.append(part_meta)
            self.id_to_path[part_id] = part_path
            self.manifest.append(part_meta.to_dict())
            if self.embeddings is not None:
                self.embeddings = np.vstack([self.embeddings, emb])
            else:
                self.embeddings = np.array([emb])
        else:
            # Already present; just update embedding and manifest.
            idx = next(i for i, pm in enumerate(self.parts) if pm.id == part_id)
            self.embeddings[idx] = emb
            self.manifest[idx] = part_meta.to_dict()
        # Update faiss index
        if self.index is not None:
            self.index.add(np.array([emb], dtype=np.float32))
        else:
            # Create new index if needed
            self.index = faiss.IndexFlatL2(len(emb))
            self.index.add(np.array(self.embeddings, dtype=np.float32))
        # Autosave
        os.makedirs("library/index", exist_ok=True)
        faiss.write_index(self.index, "library/index/faiss.index")
        with open("library/index/manifest.jsonl", "w") as f:
            for meta in self.manifest:
                f.write(json.dumps(meta) + "\n")

    @staticmethod
    def load() -> "PartIndex":
        faiss = _lazy_import_faiss()
        yaml = _lazy_import_yaml()
        if not faiss or not yaml:
            logging.warning("Could not load PartIndex due to missing dependencies.")
            return PartIndex()
        index_path = "library/index/faiss.index"
        manifest_path = "library/index/manifest.jsonl"
        pi = PartIndex()
        if os.path.exists(index_path) and os.path.exists(manifest_path):
            pi.index = faiss.read_index(index_path)
            with open(manifest_path, "r") as f:
                pi.manifest = [json.loads(line) for line in f]
            pi.parts = [PartMeta(**m) for m in pi.manifest]
            pi.id_to_path = {m["id"]: m["filename"] for m in pi.manifest}
            # Optionally, load embeddings if needed for other operations
        return pi

__all__ = [
    "PartMeta",
    "PartIndex",
    "_lazy_import_sentence_transformer",
    "_lazy_import_faiss",
    "_lazy_import_yaml",
]