import argparse
import os
import yaml

TEMPLATE = {
    "category": "unknown",
    "model": "",
    "mass": None,
    "tags": []
}

def main():
    parser = argparse.ArgumentParser(description="Generate YAML template for new part")
    parser.add_argument("--file", required=True, help="Path to part file (e.g. servo.STEP)")
    args = parser.parse_args()

    part_path = args.file
    yml_path = part_path + ".yml"

    model = os.path.splitext(os.path.basename(part_path))[0]
    tmpl = TEMPLATE.copy()
    tmpl["model"] = model

    with open(yml_path, "w") as f:
        yaml.safe_dump(tmpl, f)

    print(f"YAML template written to: {yml_path}")

if __name__ == "__main__":
    main()