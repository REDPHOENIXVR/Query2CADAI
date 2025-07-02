def build_assembly(bom, part_index):
    """
    Returns a FreeCAD macro string to import STEP parts and place them.
    """
    macro = []
    macro.append("import FreeCAD, ImportGui")
    macro.append("doc = FreeCAD.ActiveDocument")
    macro.append("if doc is None: doc = FreeCAD.newDocument()")
    # Head
    if "head" in bom:
        head_part = part_index.query("head", k=1)
        if head_part:
            macro.append(f'ImportGui.insert("{head_part[0].filename}", "Robot")')
            macro.append("FreeCAD.ActiveDocument.Objects[-1].Placement.Base = FreeCAD.Vector(0,0,60)")
    # Torso
    if "torso" in bom:
        torso_part = part_index.query("torso", k=1)
        if torso_part:
            macro.append(f'ImportGui.insert("{torso_part[0].filename}", "Robot")')
            macro.append("FreeCAD.ActiveDocument.Objects[-1].Placement.Base = FreeCAD.Vector(0,0,20)")
    # Legs
    if "legs" in bom and isinstance(bom["legs"], list):
        for idx, leg in enumerate(bom["legs"]):
            leg_part = part_index.query("leg", k=1)
            if leg_part:
                macro.append(f'ImportGui.insert("{leg_part[0].filename}", "Robot")')
                macro.append(f"FreeCAD.ActiveDocument.Objects[-1].Placement.Base = FreeCAD.Vector({-10+idx*20},0,-20)")
    # Arms
    if "arms" in bom and isinstance(bom["arms"], list):
        for idx, arm in enumerate(bom["arms"]):
            arm_part = part_index.query("arm", k=1)
            if arm_part:
                macro.append(f'ImportGui.insert("{arm_part[0].filename}", "Robot")')
                macro.append(f"FreeCAD.ActiveDocument.Objects[-1].Placement.Base = FreeCAD.Vector({-30+idx*60},40,40)")
    return "\n".join(macro)