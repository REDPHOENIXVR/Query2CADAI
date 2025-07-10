import os
import glob
import argparse
from src.logger import get_logger

try:
    import yaml
except ImportError:
    yaml = None

logger = get_logger("parts_validator")

SUPPORTED_EXTS = [".STEP", ".step", ".FCStd", ".fcstd"]

STUB_YAML = {
    "category": "unknown",
    "model": "",
    "mass": None,
    "tags": [],
    "needs_review": True
}

def ai_suggest_metadata(part_path):
    try:
        import openai, os
        if not os.environ.get("OPENAI_API_KEY"):
            return None
        with open(part_path, "rb") as f:
            content = f.read(4096)
        prompt = (
            f"This is a CAD part file named '{os.path.basename(part_path)}'. "
            "Suggest a model name and category:\n"
            "model: \ncategory: "
        )
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        text = resp.choices[0].message['content']
        lines = text.splitlines()
        model, category = "", "unknown"
        for line in lines:
            if line.lower().startswith("model:"):
                model = line.split(":", 1)[1].strip()
            elif line.lower().startswith("category:"):
                category = line.split(":", 1)[1].strip()
        return {"model": model, "category": category}
    except Exception as ex:
        logger.warning(f"AI suggestion failed for {part_path}: {ex}")
        return None

class PartValidator:
    def __init__(self, parts_dir="library/parts", auto=True, use_ai=False):
        self.parts_dir = parts_dir
        self.auto = auto
        self.use_ai = use_ai and self._ai_available()
        self.flagged = []
        self.flagged_paths = []
        self.report_lines = []
        self.stubs_created = 0

    def _ai_available(self):
        try:
            import openai, os
            return os.environ.get("OPENAI_API_KEY") is not None
        except ImportError:
            return False

    def scan(self):
        if yaml is None:
            logger.error("pyyaml is not installed.")
            return []
        files = glob.glob(os.path.join(self.parts_dir, "*"))
        flagged = []
        for f in files:
            ext = os.path.splitext(f)[1]
            if ext not in SUPPORTED_EXTS:
                continue
            yml_path = f + ".yml"
            needs_stub = False
            meta = None
            if not os.path.isfile(yml_path):
                needs_stub = True
            else:
                try:
                    with open(yml_path, "r") as ymlf:
                        meta = yaml.safe_load(ymlf) or {}
                except Exception as ex:
                    logger.warning(f"YAML load error at {yml_path}: {ex}")
                    needs_stub = True
            missing_fields = []
            if meta:
                if not meta.get("model"): missing_fields.append("model")
                if not meta.get("category"): missing_fields.append("category")
                if not meta.get("needs_review") is False:
                    missing_fields.append("needs_review")
            if needs_stub or missing_fields:
                flagged.append((f, needs_stub, missing_fields))
                self.flagged_paths.append(f)
            # Optionally auto-generate stub
            if self.auto and (needs_stub or missing_fields):
                self.generate_stub(f, yml_path, meta, needs_stub)
        self.flagged = flagged
        return flagged

    def generate_stub(self, part_path, yml_path, meta, needs_stub):
        stub = dict(STUB_YAML)
        stub["model"] = os.path.splitext(os.path.basename(part_path))[0]
        if meta and isinstance(meta, dict):
            stub.update(meta)
        if self.use_ai:
            ai_suggestion = ai_suggest_metadata(part_path)
            if ai_suggestion:
                stub.update(ai_suggestion)
        stub["needs_review"] = True
        stub.setdefault("category", "unknown")
        stub.setdefault("model", os.path.splitext(os.path.basename(part_path))[0])
        stub.setdefault("tags", [])
        stub.setdefault("mass", None)
        with open(yml_path, "w") as f:
            yaml.safe_dump(stub, f)
        self.stubs_created += 1
        logger.info(f"Stub YAML written for {os.path.basename(part_path)}")

    def report(self):
        lines = []
        if not self.flagged:
            lines.append("All parts have sufficient metadata.")
        else:
            lines.append(f"Flagged {len(self.flagged)} parts needing review:")
            for f, needs_stub, missing in self.flagged:
                base = os.path.basename(f)
                reason = []
                if needs_stub:
                    reason.append("missing YAML")
                if missing:
                    reason.append("missing fields: " + ",".join(missing))
                lines.append(f" - {base}: {'; '.join(reason)}")
        self.report_lines = lines
        print("\n".join(lines))
        # Write summary file
        os.makedirs("library/index", exist_ok=True)
        with open("library/index/flagged_parts.txt", "w") as f:
            for f_, needs_stub, missing in self.flagged:
                f.write(f"{os.path.basename(f_)}\n")

def main():
    parser = argparse.ArgumentParser(description="Scan and validate CAD part metadata.")
    parser.add_argument("--parts_dir", default="library/parts")
    parser.add_argument("--auto", action="store_true", default=True, help="Auto-create YAML stubs (default true)")
    parser.add_argument("--no-auto", dest="auto", action="store_false")
    parser.add_argument("--ai", action="store_true", help="Use OpenAI to suggest metadata if available")
    args = parser.parse_args()

    v = PartValidator(parts_dir=args.parts_dir, auto=args.auto, use_ai=args.ai)
    v.scan()
    v.report()

if __name__ == "__main__":
    main()