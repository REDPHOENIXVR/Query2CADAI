import argparse
import os
import logging

from src import vision_bom, skeleton, assembly_builder, parts_index

def main():
    parser = argparse.ArgumentParser(description="One-click CLI: image → BOM → skeleton → assembly")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--outdir", required=True, help="Directory for output files")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Step a: extract BOM from image
    bom = vision_bom.extract_bom(args.image)
    print("Extracted BOM:")
    for entry in bom:
        print(f"  {entry['qty']}x {entry['category']} {entry['model']}")

    # Step b: generate skeleton macro
    skeleton_macro = skeleton.generate_skeleton(bom)
    skeleton_macro_path = os.path.join(args.outdir, "skeleton.FCMacro")
    with open(skeleton_macro_path, "w") as f:
        f.write(skeleton_macro)
    print(f"Skeleton macro written to: {skeleton_macro_path}")

    # Step c: build parts index if needed and build assembly
    idx = parts_index.PartIndex(index_path="data/parts")
    idx.build_index()
    assembly_macro = assembly_builder.build_assembly(bom, idx)
    assembly_macro_path = os.path.join(args.outdir, "assembly.FCMacro")
    with open(assembly_macro_path, "w") as f:
        f.write(assembly_macro)
    print(f"Assembly macro written to: {assembly_macro_path}")

if __name__ == "__main__":
    main()