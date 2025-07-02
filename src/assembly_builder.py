def build_assembly(bom, part_index):
    """
    Returns a FreeCAD macro string to import STEP parts and place them.
    Falls back to simple primitive geometry if retrieval fails.
    """
    macro = []
    macro.append("import FreeCAD, ImportGui, Part, Draft")
    macro.append("doc = FreeCAD.ActiveDocument")
    macro.append("if doc is None: doc = FreeCAD.newDocument()")
    # Head
    head_done = False
    if "head" in bom:
        head_part = part_index.query("head", k=1)
        if head_part:
            macro.append(f'ImportGui.insert("{head_part[0].filename}", "Robot")')
            macro.append("FreeCAD.ActiveDocument.Objects[-1].Placement.Base = FreeCAD.Vector(0,0,60)")
            head_done = True
        # Fallback: use sphere if nothing found
        if not head_part:
            macro.append("head = Part.makeSphere(20)")
            macro.append("Part.show(head)")
            macro.append("FreeCAD.ActiveDocument.Objects[-1].Placement.Base = FreeCAD.Vector(0,0,60)")
            head_done = True
    # Torso
    torso_done = False
    if "torso" in bom:
        torso_part = part_index.query("torso", k=1)
        if torso_part:
            macro.append(f'ImportGui.insert("{torso_part[0].filename}", "Robot")')
            macro.append("FreeCAD.ActiveDocument.Objects[-1].Placement.Base = FreeCAD.Vector(0,0,20)")
            torso_done = True
        if not torso_part:
            macro.append("torso = Part.makeBox(40,60,30)")
            macro.append("Part.show(torso)")
            macro.append("FreeCAD.ActiveDocument.Objects[-1].Placement.Base = FreeCAD.Vector(0,0,20)")
            torso_done = True
    # Legs
    if "legs" in bom and isinstance(bom["legs"], list):
        for idx, leg in enumerate(bom["legs"]):
            leg_part = part_index.query("leg", k=1)
            if leg_part:
                macro.append(f'ImportGui.insert("{leg_part[0].filename}", "Robot")')
                macro.append(f"FreeCAD.ActiveDocument.Objects[-1].Placement.Base = FreeCAD.Vector({-10+idx*20},0,-20)")
            else:
                macro.append(f"leg{idx} = Part.makeCylinder(8, 60, FreeCAD.Vector({-10+idx*20},0,-60))")
                macro.append(f"Part.show(leg{idx})")
                macro.append(f"FreeCAD.ActiveDocument.Objects[-1].Placement.Base = FreeCAD.Vector({-10+idx*20},0,-20)")
    # Arms
    if "arms" in bom and isinstance(bom["arms"], list):
        for idx, arm in enumerate(bom["arms"]):
            arm_part = part_index.query("arm", k=1)
            if arm_part:
                macro.append(f'ImportGui.insert("{arm_part[0].filename}", "Robot")')
                macro.append(f"FreeCAD.ActiveDocument.Objects[-1].Placement.Base = FreeCAD.Vector({-30+idx*60},40,40)")
            else:
                macro.append(f"arm{idx} = Part.makeCylinder(6, 50, FreeCAD.Vector({-30+idx*60},40,10))")
                macro.append(f"Part.show(arm{idx})")
                macro.append(f"FreeCAD.ActiveDocument.Objects[-1].Placement.Base = FreeCAD.Vector({-30+idx*60},40,40)")
    return "\n".join(macro)