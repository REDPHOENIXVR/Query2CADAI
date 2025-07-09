import argparse, os, json
from datetime import datetime
from src.vision_bom import extract_bom
from src.skeleton import generate_skeleton
from src.parts_index import PartIndex
from src.assembly_builder import build_assembly


def main():
    parser = argparse.ArgumentParser(description="Image → BOM → skeleton & assembly macros")
    parser.add_argument("--image", type=str, help="Path to input image file (PNG/JPG)")
    parser.add_argument("--text", type=str, help="Text prompt for concept image (alternative to --image)")
    parser.add_argument("--outdir", required=True, help="Output directory")
    args = parser.parse_args()

    # --text or --image logic
    if args.text and not args.image:
        from src.image_generator import generate_image
        print(f"No --image provided. Generating image from prompt: {args.text!r}")
        image_dir = os.path.join(args.outdir, "images")
        image_path = generate_image(args.text, outdir=image_dir)
        # Save as generated.png for user visibility
        generated_path = os.path.join(image_dir, "generated.png")
        try:
            import shutil
            shutil.copy(image_path, generated_path)
            image_path = generated_path
        except Exception as e:
            print(f"Could not save generated image as generated.png: {e}")
    elif args.text and args.image:
        print("Note: --image provided, ignoring --text and using explicit image file.")
        image_path = args.image
    elif args.image:
        image_path = args.image
    else:
        parser.error("You must provide either --image or --text (one required).")

    os.makedirs(args.outdir, exist_ok=True)

    # 1. BOM
    bom = extract_bom(image_path)
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