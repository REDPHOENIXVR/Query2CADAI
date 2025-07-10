import os
import tempfile
import shutil

def test_parts_validator_stub_creation():
    # Import here to avoid global import error if pyyaml not present
    from src.parts_validator import PartValidator
    import yaml
    with tempfile.TemporaryDirectory() as tmpdir:
        # Simulate a new STEP file
        part_path = os.path.join(tmpdir, "test_part.STEP")
        with open(part_path, "w") as f:
            f.write("dummy step content")
        validator = PartValidator(parts_dir=tmpdir, auto=True, use_ai=False)
        flagged = validator.scan()
        yml_path = part_path + ".yml"
        assert os.path.exists(yml_path), "YAML stub not created"
        with open(yml_path, "r") as f:
            meta = yaml.safe_load(f)
        assert meta.get("needs_review", False) == True, "needs_review not set to true"
        assert meta.get("category") == "unknown"
        assert meta.get("model") == "test_part"
        assert isinstance(meta.get("tags"), list)
        print("test_parts_validator_stub_creation passed.")

if __name__ == "__main__":
    test_parts_validator_stub_creation()