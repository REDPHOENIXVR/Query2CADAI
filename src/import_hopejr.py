import argparse
import os
import shutil
import glob
import logging
import pathlib
import sys
import yaml

# Replicates PartIndex._ensure_minimal_yaml logic
def ensure_yaml(stp_path):
    yml_path = os.path.splitext(stp_path)[0] + ".yml"
    if os.path.exists(yml_path):
        return
    stub = {
        "category": "unknown",
        "model": pathlib.Path(stp_path).stem,
        "mass": None,
        "tags": [],
        "needs_review": True,
    }
    with open(yml_path, "w") as f:
        yaml.safe_dump(stub, f)

def main():
    parser = argparse.ArgumentParser(description="Import HOPEJr STEP files into the Query2CAD part library")
    parser.add_argument("--hopejr-dir", default="external/HOPEJr", help="Path to cloned HOPEJr repo")
    parser.add_argument("--dest", default="library/parts", help="Destination parts directory")
    args = parser.parse_args()
    src_dir = os.path.join(args.hopejr_dir, "Humanoid", "STEP")
    if not os.path.isdir(src_dir):
        logging.error("STEP directory not found in HOPEJr repo: %s", src_dir)
        sys.exit(1)
    os.makedirs(args.dest, exist_ok=True)
    cnt = 0
    for stp in glob.glob(os.path.join(src_dir, "*.STEP")):
        dest_path = os.path.join(args.dest, f"hopejr_{os.path.basename(stp)}")
        if not os.path.exists(dest_path):
            shutil.copy2(stp, dest_path)
            ensure_yaml(dest_path)
            cnt += 1
    print(f"Imported {cnt} new STEP files from HOPEJr.")

if __name__ == "__main__":
    main()