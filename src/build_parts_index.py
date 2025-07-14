from src.parts_index import PartIndex
from src.parts_validator import PartValidator
from src.thumbnail import generate_thumbnail
import os

if __name__ == "__main__":
    validator = PartValidator()
    validator.scan()
    validator.report()

    # Generate thumbnails for each part if not already present
    parts_dir = "library/parts"
    thumbnails_dir = "library/thumbnails"
    os.makedirs(thumbnails_dir, exist_ok=True)
    for fname in os.listdir(parts_dir):
        if not fname.lower().endswith((".step", ".stp", ".fcstd")):
            continue
        part_path = os.path.join(parts_dir, fname)
        generate_thumbnail(part_path, outdir=thumbnails_dir)

    PartIndex().build_index()