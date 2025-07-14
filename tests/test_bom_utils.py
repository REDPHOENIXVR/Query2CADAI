import pytest
from src.bom_utils import validate_bom

def test_validate_bom_missing_fields():
    # Only provide 'head', omit 'legs' and 'torso'
    bom = {"head": {"type": "sensor", "material": "plastic"}}
    corrected, warnings = validate_bom(bom)

    # Should add torso (default), and two legs (left/right, default)
    assert "torso" in corrected
    assert "legs" in corrected
    assert isinstance(corrected["legs"], list)
    assert len(corrected["legs"]) == 2
    leg_sides = sorted([leg.get("side") for leg in corrected["legs"]])
    assert leg_sides == ["left", "right"]
    # Torso should be default
    assert corrected["torso"]["type"] == "unknown"
    assert corrected["torso"]["material"] == "unknown"
    # Should emit warnings (at least for missing keys and legs)
    assert len(warnings) > 0
    assert any("Missing 'torso'" in w for w in warnings)
    assert any("Missing 'legs'" in w for w in warnings)
    assert any("No legs present" in w for w in warnings)

def test_validate_bom_single_leg():
    # Provide a single leg, missing side
    bom = {
        "head": {"type": "core", "material": "steel"},
        "torso": {"type": "solid", "material": "carbon"},
        "legs": [{"type": "hydraulic", "material": "iron"}]
    }
    corrected, warnings = validate_bom(bom)
    # Should have two legs with sides left/right
    assert "legs" in corrected
    assert len(corrected["legs"]) == 2
    sides = sorted([leg.get("side") for leg in corrected["legs"]])
    assert sides == ["left", "right"]
    # Both legs should have "hydraulic" as type (since it was duplicated)
    assert all(leg["type"] == "hydraulic" for leg in corrected["legs"])
    # Should warn about duplication
    assert any("Only one leg present" in w for w in warnings)