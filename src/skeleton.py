def generate_skeleton(bom, param=None):
    """
    Generates a FreeCAD macro as a string for basic humanoid skeleton geometry, scaled by parametric sizes.
    Optionally, if the BOM dict contains 'joints', creates visual spheres at joint coordinates.

    Args:
        bom (dict): BOM dictionary with possible keys: "head", "torso", "legs", "arms", and optional "joints".
        param (dict, optional): {"height", "leg_length", "arm_length"} in cm.

    Returns:
        str: FreeCAD macro string.
    """
    # Defaults
    defaults = {"height": 180.0, "leg_length": 70.0, "arm_length": 60.0}
    if param is None:
        param = defaults
    else:
        # Fill in any missing keys with defaults
        param = {**defaults, **(param or {})}

    # Extract and clamp values
    height_cm = float(param.get("height", 180.0))
    leg_len_cm = float(param.get("leg_length", 70.0))
    arm_length_cm = float(param.get("arm_length", 60.0))
    # Estimate torso height
    torso_height = max(height_cm - leg_len_cm - 30, 30)
    head_radius = height_cm * 0.06
    torso_width = height_cm * 0.22
    torso_depth = height_cm * 0.12
    leg_radius = 8.0 * (leg_len_cm / 70.0)  # scale leg thickness a bit
    arm_radius = 6.0 * (arm_length_cm / 60.0)  # scale arm thickness a bit

    macro = []
    macro.append("import FreeCAD, Part, Draft")
    macro.append("doc = FreeCAD.ActiveDocument")
    macro.append("if doc is None: doc = FreeCAD.newDocument()")
    macro.append(f"# === Parametric Skeleton Variables ===")
    macro.append(f"height_cm = {height_cm}")
    macro.append(f"leg_len_cm = {leg_len_cm}")
    macro.append(f"arm_length_cm = {arm_length_cm}")
    macro.append(f"torso_height = {torso_height}")
    macro.append(f"head_radius = {head_radius}")
    macro.append(f"torso_width = {torso_width}")
    macro.append(f"torso_depth = {torso_depth}")
    macro.append(f"leg_radius = {leg_radius}")
    macro.append(f"arm_radius = {arm_radius}")
    macro.append("")

    # Placements:
    # Legs: at z=0 (feet on ground)
    # Torso: base at z=leg_len_cm
    # Head: center at z=(torso_height/2 + leg_len_cm + head_radius)
    # Arms: attach at height z=(torso_height + leg_len_cm - arm_length_cm*0.3)

    # Head
    if "head" in bom:
        head_z = torso_height + leg_len_cm + head_radius  # center of head sphere
        macro.append(f"head = Part.makeSphere(head_radius, FreeCAD.Vector(0,0,{head_z}))")
        macro.append("Part.show(head)")
    # Torso
    if "torso" in bom:
        # Place center of base at (0,0,leg_len_cm), so box extends up
        macro.append(
            f"torso = Part.makeBox(torso_width, torso_depth, torso_height, FreeCAD.Vector(-torso_width/2, -torso_depth/2, leg_len_cm))"
        )
        macro.append("Part.show(torso)")
    # Legs
    if "legs" in bom and isinstance(bom["legs"], list):
        n_legs = len(bom["legs"])
        spacing = torso_width * 0.4 if n_legs > 1 else 0
        for idx, leg in enumerate(bom["legs"]):
            # Spread legs left/right (x axis)
            if n_legs == 1:
                x = 0
            else:
                x = -spacing/2 + idx * (spacing/(n_legs-1)) if n_legs > 1 else 0
            macro.append(
                f"leg{idx} = Part.makeCylinder(leg_radius, leg_len_cm, FreeCAD.Vector({x}, 0, 0))"
            )
            macro.append(f"Part.show(leg{idx})")
    # Arms
    if "arms" in bom and isinstance(bom["arms"], list):
        n_arms = len(bom["arms"])
        y_attach = torso_depth/2 + arm_radius + 2
        z_attach = leg_len_cm + torso_height * 0.7  # attach upper on torso
        for idx, arm in enumerate(bom["arms"]):
            # Spread arms left/right (x axis)
            if n_arms == 1:
                x = 0
            else:
                x = -torso_width/2 + idx * (torso_width/(n_arms-1)) if n_arms > 1 else 0
            macro.append(
                f"arm{idx} = Part.makeCylinder(arm_radius, arm_length_cm, FreeCAD.Vector({x}, {y_attach}, {z_attach}))"
            )
            macro.append(f"Part.show(arm{idx})")

    # Joints (optional)
    joints = bom.get("joints")
    if joints and isinstance(joints, list):
        macro.append("# --- Visualize joints as spheres ---")
        for idx, joint in enumerate(joints):
            # Expecting {"x":..., "y":..., "z":...} in millimeters
            try:
                x = float(joint["x"])
                y = float(joint["y"])
                z = float(joint["z"])
            except (KeyError, TypeError, ValueError):
                continue  # skip malformed joint entry
            macro.append(
                f"joint_sphere_{idx} = Part.makeSphere(4.0, FreeCAD.Vector({x}, {y}, {z}))"
            )
            macro.append(
                f"Draft.setColor(joint_sphere_{idx}, (1.0, 0.2, 0.2))"
            )
            macro.append(f"Part.show(joint_sphere_{idx})")
    macro.append("FreeCAD.ActiveDocument.recompute()")
    return "\n".join(macro)