import argparse, os, json
from datetime import datetime
from src.vision_bom import extract_bom
from src.skeleton import generate_skeleton
from src.parts_index import PartIndex
from src.assembly_builder import build_assembly


def main():
    parser = argparse.ArgumentParser(description="Image → BOM → skeleton & assembly macros")
    parser.add_argument("--image", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 1. BOM
    bom = extract_bom(args.image)
    bom_path = os.path.join(args.outdir, "bom.json")
    with open(bom_path, "w") as f:
        json.dump(bom, f, indent=2)
    print(f"BOM saved to {bom_path}")

    # 2. Skeleton macro
    sk_code = generate_skeleton(bom)
    sk_path = os.path.join(args.outdir, "skeleton.FCMacro")
    with open(sk_path, "w") as f:
        f.write(sk_code)
    print(f"Skeleton macro → {sk_path}")

    # 3. Parts index & assembly macro
    idx = PartIndex()
    if idx.index is None or len(idx.parts) == 0:
        print("Building parts index …")
        idx.build_index("library/parts")
    asm_code = build_assembly(bom, idx)
    asm_path = os.path.join(args.outdir, "assembly.FCMacro")
    with open(asm_path, "w") as f:
        f.write(asm_code)
    print(f"Assembly macro → {asm_path}")

    print("Done ✅")

if __name__ == "__main__":
    main()