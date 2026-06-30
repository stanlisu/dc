"""Codec round-trip, bijection, and structure-preservation tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from obfuscation.codec import Codec

MAP = Path(__file__).resolve().parent.parent / "map.json"


@pytest.fixture(scope="module")
def codec():
    return Codec(MAP)


@pytest.fixture(scope="module")
def data():
    return json.loads(MAP.read_text())


def test_map_bijective(data):
    for ns in ("regimes", "features"):
        names = list(data[ns].keys())
        codes = list(data[ns].values())
        assert len(names) == len(set(names))
        assert len(codes) == len(set(codes))


def test_codes_are_opaque_and_structure_safe(data):
    # Codes must not contain structural tokens or look like TF prefixes.
    import re
    tf = re.compile(r"^\d+[smhd]_")
    for ns in ("regimes", "features"):
        for code in data[ns].values():
            assert "_and_" not in code
            assert not code.endswith(("_long", "_short"))
            assert not tf.match(code)


def test_atom_roundtrip_all(codec, data):
    for name in data["regimes"]:
        assert codec.decode_atom(codec.encode_atom(name)) == name
    for name in data["features"]:
        assert codec.decode_feature_base(codec.encode_feature_base(name)) == name


def test_unknown_raises(codec):
    with pytest.raises(KeyError):
        codec.encode_atom("definitely_not_a_regime")
    with pytest.raises(KeyError):
        codec.decode_atom("r999")
    with pytest.raises(KeyError):
        codec.encode_feature_base("definitely_not_a_feature")


def test_regime_structure_preserved(codec, data):
    atoms = list(data["regimes"])
    a, b = atoms[0], atoms[1]
    # composite cross-TF with position
    name = f"1d_{a}_and_1h_{b}_long"
    enc = codec.encode_regime(name)
    assert enc.startswith("1d_") and "_and_1h_" in enc and enc.endswith("_long")
    assert codec.encode_atom(a) in enc and codec.encode_atom(b) in enc
    assert codec.decode_regime(enc) == name


def test_regime_simple_and_positionless(codec, data):
    a = next(iter(data["regimes"]))
    assert codec.decode_regime(codec.encode_regime(a)) == a               # bare atom
    assert codec.decode_regime(codec.encode_regime(f"{a}_short")) == f"{a}_short"


def test_feature_structure_preserved(codec, data):
    f = next(iter(data["features"]))
    assert codec.decode_feature(codec.encode_feature(f)) == f             # no TF prefix
    assert codec.decode_feature(codec.encode_feature(f"15m_{f}")) == f"15m_{f}"


def test_tolerant_decode_accepts_code_and_real(codec, data):
    atoms = list(data["regimes"])
    a, b = atoms[0], atoms[1]
    real = f"1d_{a}_and_1h_{b}_long"
    coded = codec.encode_regime(real)
    # coded -> real
    assert codec.decode_regime_tolerant(coded) == real
    # already-real -> unchanged (mixed namespaces during rollout)
    assert codec.decode_regime_tolerant(real) == real
    # mixed (one coded, one real) resolves both to real
    mixed = f"1d_{codec.encode_atom(a)}_and_1h_{b}_long"
    assert codec.decode_regime_tolerant(mixed) == real


def test_or_regime_roundtrip(codec, data):
    a, b = list(data["regimes"])[:2]
    name = f"{a}_or_{b}"
    enc = codec.encode_regime(name)
    assert "_or_" in enc and codec.encode_atom(a) in enc and codec.encode_atom(b) in enc
    assert codec.decode_regime(enc) == name
    # mixed _and_/_or_ with TF prefixes + position
    mixed = f"1d_{a}_and_1h_{b}_or_4h_{a}_long"
    assert codec.decode_regime(codec.encode_regime(mixed)) == mixed


def test_has_baseline(codec, data):
    a = next(iter(data["regimes"]))
    assert codec.has_baseline("baseline")
    assert codec.has_baseline("1d_baseline_long")
    assert codec.has_baseline(f"{a}_and_baseline")
    assert codec.has_baseline(f"{a}_or_4h_baseline")
    assert not codec.has_baseline(a)
    assert not codec.has_baseline(codec.encode_regime(a))   # coded never baseline


def test_no_short_long_false_split(codec, data):
    # An atom whose name merely ends in a non-position word must not be mangled.
    for name in data["regimes"]:
        assert codec.decode_atom(codec.encode_atom(name)) == name
