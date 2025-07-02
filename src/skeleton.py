def generate_skeleton(bom, param=None):
    # Generates FreeCAD macro as a string placing basic geometry
    macro = []
    macro.append("import FreeCAD, Part, Draft")
    macro.append("doc = FreeCAD.ActiveDocument")
    macro.append("if doc is None: doc = FreeCAD.newDocument()")
    # Head
    if "head" in bom:
        macro.append("head = Part.makeSphere(20)")
        macro.append("Part.show(head)")
    if "torso" in bom:
        macro.append("torso = Part.makeBox(40,60,30)")
        macro.append("Part.show(torso)")
    # Legs
    if "legs" in bom and isinstance(bom["legs"], list):
        for idx, leg in enumerate(bom["legs"]):
            macro.append(f"leg{idx} = Part.makeCylinder(8, 60, FreeCAD.Vector({-10+idx*20},0,-60))")
            macro.append(f"Part.show(leg{idx})")
    # Arms
    if "arms" in bom and isinstance(bom["arms"], list):
        for idx, arm in enumerate(bom["arms"]):
            macro.append(f"arm{idx} = Part.makeCylinder(6, 50, FreeCAD.Vector({-30+idx*60},40,10))")
            macro.append(f"Part.show(arm{idx})")
    macro.append("FreeCAD.ActiveDocument.recompute()")
    return "\n".join(macro)