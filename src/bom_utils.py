from typing import Optional, List, Dict, Tuple, Any
from pydantic import BaseModel, Field, ValidationError, root_validator
import copy

class Head(BaseModel):
    type: str = "unknown"
    material: Optional[str] = "unknown"

class Torso(BaseModel):
    type: str = "unknown"
    material: Optional[str] = "unknown"

class Limb(BaseModel):
    type: str = "unknown"
    material: Optional[str] = "unknown"
    side: Optional[str] = None

def validate_bom(bom: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """
    Validate and auto-correct a BOM dict using Pydantic schemas.

    Returns corrected bom and a list of warning strings describing what was fixed.
    """
    warnings = []
    corrected = copy.deepcopy(bom) if bom else {}

    # Ensure required keys
    for key in ['head', 'torso', 'legs']:
        if key not in corrected:
            warnings.append(f"Missing '{key}' key; default inserted.")
            if key == 'legs':
                corrected[key] = []
            else:
                corrected[key] = {}

    # Head
    head = corrected.get('head', {})
    try:
        corrected['head'] = Head(**head).dict()
    except ValidationError as e:
        warnings.append(f"Head fields invalid or missing; using defaults. Details: {e}")
        corrected['head'] = Head().dict()

    # Torso
    torso = corrected.get('torso', {})
    try:
        corrected['torso'] = Torso(**torso).dict()
    except ValidationError as e:
        warnings.append(f"Torso fields invalid or missing; using defaults. Details: {e}")
        corrected['torso'] = Torso().dict()

    # Limbs
    def validate_limb_list(lname, limbs):
        fixed_limbs = []
        has_invalid = False
        for l in limbs if isinstance(limbs, list) else []:
            try:
                fixed_limbs.append(Limb(**l).dict())
            except ValidationError:
                has_invalid = True
                fixed_limbs.append(Limb().dict())
        if has_invalid:
            warnings.append(f"Some {lname} entries invalid; fixed with defaults.")
        return fixed_limbs

    # Legs
    legs = corrected.get('legs', [])
    if not isinstance(legs, list):
        warnings.append("Legs field not a list; resetting to empty list.")
        legs = []
    fixed_legs = validate_limb_list('legs', legs)
    # Guarantee at least 2 legs
    if len(fixed_legs) == 0:
        warnings.append("No legs present; adding left/right unknown legs.")
        fixed_legs = [
            Limb(side="left").dict(),
            Limb(side="right").dict()
        ]
    elif len(fixed_legs) == 1:
        # Duplicate and assign sides
        warnings.append("Only one leg present; duplicating to have left/right.")
        side0 = fixed_legs[0].get("side")
        fixed_legs[0]["side"] = "left"
        dup_leg = copy.deepcopy(fixed_legs[0])
        dup_leg["side"] = "right"
        fixed_legs = [fixed_legs[0], dup_leg]
    elif len(fixed_legs) == 2:
        # Ensure sides assigned
        if not fixed_legs[0].get("side"):
            fixed_legs[0]["side"] = "left"
        if not fixed_legs[1].get("side"):
            fixed_legs[1]["side"] = "right"
    corrected['legs'] = fixed_legs

    # Arms (optional)
    arms = corrected.get('arms', None)
    if arms is not None:
        if not isinstance(arms, list):
            warnings.append("Arms field not a list; resetting to empty list.")
            arms = []
        fixed_arms = validate_limb_list('arms', arms)
        # Assign sides if only two arms and missing
        if len(fixed_arms) == 2:
            if not fixed_arms[0].get("side"):
                fixed_arms[0]["side"] = "left"
            if not fixed_arms[1].get("side"):
                fixed_arms[1]["side"] = "right"
        corrected['arms'] = fixed_arms

    return corrected, warnings