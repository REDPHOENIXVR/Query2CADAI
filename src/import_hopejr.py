import argparse
import os
import shutil
import glob
import logging
import pathlib
import sys
import yaml

def ensure_yaml(stp_path):
    yml_path = os.path.splitext(stp_path)[0] + ".yml"
    if os.path.exists(yml_path):
        return False
    stem = pathlib.Path(stp_path).stem.lower()
    if "arm" in stem:
        category = "arm assembly"
    elif "leg" in stem:
        category = "leg assembly"
    else:
        category = "assembly"
    stub = {
        "category": category,
        "model": pathlib.Path(stp_path).stem,
        "mass": None,
        "tags": ["hopejr"],
        "needs_review": True,
    }
    with open(yml_path, "w") as f:
        yaml.safe_dump(stub, f)
    return True

def main():
    parser = argparse.ArgumentParser(description="Import HOPEJr STEP files into the Query2CAD part library")
    parser.add_argument("--hopejr-dir", default="external/HOPEJr", help="Path to cloned HOPEJr repo")
    parser.add_argument("--dest", default="library/parts", help="Destination parts directory")
    parser.add_argument("--run-validator", dest="run_validator", action="store_true", default=True,
                        help="Run AI-powered part validator after import (default: True)")
    parser.add_argument("--no-validator", dest="run_validator", action="store_false",
                        help="Disable part validator after import")
    parser.add_argument("--run-index", dest="run_index", action="store_true", default=True,
                        help="Rebuild the parts index after import (default: True)")
    parser.add_argument("--no-index", dest="run_index", action="store_false",
                        help="Disable index rebuild after import")
    args = parser.parse_args()
    src_dir = os.path.join(args.hopejr_dir, "Humanoid", "STEP")
    if not os.path.isdir(src_dir):
        logging.error("STEP directory not found in HOPEJr repo: %s", src_dir)
        sys.exit(1)
    os.makedirs(args.dest, exist_ok=True)
    copied = 0
    stubbed = 0
    for stp in glob.glob(os.path.join(src_dir, "*.STEP")):
        dest_path = os.path.join(args.dest, f"hopejr_{os.path.basename(stp)}")
        if not os.path.exists(dest_path):
            shutil.copy2(stp, dest_path)
            copied += 1
            if ensure_yaml(dest_path):
                stubbed += 1
    print(f"Imported {copied} new STEP files from HOPEJr.")
    print(f"Stubbed metadata for {stubbed} files.")

    validated = None
    flagged = None
    indexed = None
    if args.run_validator:
        try:
            from src.parts_validator import PartValidator
            v = PartValidator(parts_dir=args.dest, use_ai=True)
            flagged = v.scan()
            v.report()
            validated = len(flagged)
        except Exception as e:
            print("Validator failed:", e)
    if args.run_index:
        try:
            from src.parts_index import PartIndex
            PartIndex().build_index(library_dir=args.dest)
            indexed = True
        except Exception as e:
            print("Indexing failed:", e)

    print("\n=== HOPEJr Import Summary ===")
    print(f"Files copied: {copied}")
    print(f"Metadata stubbed: {stubbed}")
    if validated is not None:
        print(f"Parts validated: {validated}")
    if indexed:
        print("Parts index rebuilt.")
    print("Done.\n")

if __name__ == "__main__":
    main()