"""The TF ladder became configurable (2026-09-24); the DEFAULT stack must not move.

`OrbResearch.generate_regime_stack()` used to build sections 2, 4 and 5 from
~137 hand-written tuples naming '1d_', '4h_' and '1h_' as literals. Those tables
are now rank-based templates over an ordered ladder, so the same code generates
a 1m/5m/15m/1h stack. The rewrite is only safe if the legacy ladder still emits
the SHIPPED stack exactly — same names, same positions, same order, same count.

`fixtures/regime_stack_default_ladder.json` was captured by RUNNING the
pre-refactor code on origin/main (c7c0ef6) and is pinned here. It is never
regenerated from the current implementation: an expectation recomputed by the
thing under test proves nothing.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, "agamotto_pkg/src")
sys.path.insert(0, "orb_pkg/src")
sys.path.insert(0, ".")

from orb.research import (  # noqa: E402
    _DEFAULT_TIMEFRAMES,
    _resolve_ladder,
    _tf_at_rank,
    OrbResearch,
)

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "regime_stack_default_ladder.json"

# A TF prefix at the start of a name or of an `_and_` conjunct. Regime CODES are
# `r<digits>`, which start with a letter, so this never matches inside one.
_TF_TOKEN = re.compile(r"(?<![0-9a-zA-Z])(\d+[smhd])_")

_NEW_LADDER = ["1m", "5m", "15m", "1h"]


def _pinned() -> list[dict]:
    return json.loads(_FIXTURE.read_text())


def test_zero_arg_stack_is_byte_identical_to_the_shipped_stack():
    assert OrbResearch.generate_regime_stack() == _pinned()


def test_pinned_fixture_is_the_332_row_stack():
    """Guards the fixture itself against being quietly swapped for a rerun."""
    rows = _pinned()
    assert len(rows) == 332
    assert sum(1 for r in rows if "_and_" not in r["regime"]) == 172
    assert sum(1 for r in rows if "_and_" in r["regime"]) == 160


def test_explicit_legacy_timeframes_matches_the_zero_arg_call():
    cfg = {"TIMEFRAMES": list(_DEFAULT_TIMEFRAMES)}
    assert OrbResearch.generate_regime_stack(cfg) == _pinned()


def test_config_without_timeframes_key_matches_the_zero_arg_call():
    assert OrbResearch.generate_regime_stack({"SYMBOLS": ["X"]}) == _pinned()


def test_new_ladder_is_the_same_table_with_the_timeframes_substituted():
    """A 1m/5m/15m/1h arm gets the SAME regime structure, re-addressed.

    Rank-for-rank: 1d->1h, 4h->15m, 1h->5m, 15m->1m. If any template still
    carried a literal timeframe, the substituted default would not match.
    """
    subst = {
        old: new
        for old, new in zip(_DEFAULT_TIMEFRAMES, _NEW_LADDER)
    }
    expected = [
        {
            "regime": _TF_TOKEN.sub(lambda m: f"{subst[m.group(1)]}_", r["regime"]),
            "position": r["position"],
        }
        for r in _pinned()
    ]
    got = OrbResearch.generate_regime_stack({"TIMEFRAMES": list(_NEW_LADDER)})
    assert got == expected


def test_new_ladder_leaks_no_legacy_timeframe():
    got = OrbResearch.generate_regime_stack({"TIMEFRAMES": list(_NEW_LADDER)})
    present = {m for r in got for m in _TF_TOKEN.findall(r["regime"])}
    assert present <= set(_NEW_LADDER), present
    assert "4h" not in present and "1d" not in present


def test_rank_zero_is_the_coarsest_timeframe():
    assert _tf_at_rank(_DEFAULT_TIMEFRAMES, 0) == "1d"
    assert _tf_at_rank(_DEFAULT_TIMEFRAMES, 3) == "15m"
    with pytest.raises(IndexError):
        _tf_at_rank(["1m", "5m"], 2)


def test_misordered_ladder_raises_rather_than_building_it_backwards():
    with pytest.raises(ValueError, match="finest-first"):
        _resolve_ladder({"TIMEFRAMES": ["1d", "4h", "1h", "15m"]})


def test_empty_duplicate_and_unparseable_ladders_raise():
    with pytest.raises(ValueError, match="non-empty"):
        _resolve_ladder({"TIMEFRAMES": []})
    with pytest.raises(ValueError, match="duplicate"):
        _resolve_ladder({"TIMEFRAMES": ["15m", "15m"]})
    with pytest.raises(ValueError, match="unparseable"):
        _resolve_ladder({"TIMEFRAMES": ["15minutes", "1h"]})


def test_ladder_too_short_for_the_cross_tf_templates_raises():
    """The templates reach rank 2; a two-rung ladder cannot satisfy them."""
    with pytest.raises(IndexError, match="rank 2"):
        OrbResearch.generate_regime_stack({"TIMEFRAMES": ["1m", "5m"]})
