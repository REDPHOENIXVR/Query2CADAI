import os

def generate_thumbnail(part_path: str, outdir: str = "library/thumbnails") -> str:
    """
    Generate a 256x256 PNG thumbnail for the given CAD part.
    Uses FreeCAD if available; falls back to a gray placeholder if rendering fails.
    Returns the path to the generated thumbnail.
    """
    os.makedirs(outdir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(part_path))[0]
    out_png = os.path.join(outdir, f"{basename}.png")

    # If thumbnail already exists, return its path
    if os.path.isfile(out_png):
        return out_png

    try:
        import FreeCAD
        import FreeCADGui
        import Part
        App = FreeCAD
        Gui = FreeCADGui

        # Open document
        doc = App.openDocument(part_path)
        Gui.showMainWindow()
        Gui.activateWorkbench("PartWorkbench")
        Gui.activeDocument().activeView().viewIsometric()
        Gui.activeDocument().activeView().fitAll()
        Gui.activeDocument().activeView().saveImage(out_png, 256, 256, 'White')
        Gui.activeDocument().close()
        if os.path.isfile(out_png):
            return out_png
    except Exception:
        pass  # Fall back to placeholder

    # Fallback: Create a solid gray placeholder using Pillow
    try:
        from PIL import Image
        img = Image.new("RGB", (256, 256), (180, 180, 180))
        img.save(out_png)
        return out_png
    except Exception:
        # Could not create a fallback; return empty string
        return ""