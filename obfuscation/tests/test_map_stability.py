"""build_map must NEVER renumber: deployed artifacts (coded weights meta,
filter parquet columns, stack CSVs) reference codes by value.

Regression for 2026-07-24: sequential enumeration renumbered every code after
an inventory removal (dropping oi_velocity/oi_acceleration shifted
r053..r072 / f062..f100) — and conftest re-runs build_map every test session.
"""
import importlib.util
import json
from pathlib import Path

_BUILD_MAP = Path(__file__).resolve().parent.parent / "build_map.py"
_spec = importlib.util.spec_from_file_location("build_map_under_test", _BUILD_MAP)
build_map = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_map)


def test_existing_codes_preserved_on_removal_and_addition():
    existing = {"alpha": "r001", "beta": "r002", "gamma": "r003"}
    # beta removed from inventory, delta added
    merged = build_map._stable_assign(["alpha", "delta", "gamma"], existing, "r")
    assert merged["alpha"] == "r001"
    assert merged["gamma"] == "r003"          # NOT renumbered
    assert merged["beta"] == "r002"           # retired name keeps its code
    assert merged["delta"] == "r004"          # new name appends, never reuses r002


def test_committed_map_is_fixed_point():
    """Running build_map over the committed inventory must reproduce the
    committed map byte-for-byte (i.e. the repo is never left renumbered)."""
    here = Path(__file__).resolve().parent.parent
    inv = json.loads((here / "inventory.json").read_text())
    prev = json.loads((here / "map.json").read_text())
    regimes = build_map._stable_assign(sorted(inv["regime_atoms"]), prev["regimes"], "r")
    features = build_map._stable_assign(sorted(inv["feature_cols"]), prev["features"], "f")
    assert regimes == prev["regimes"]
    assert features == prev["features"]
