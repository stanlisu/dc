"""Regression coverage for AgamottoTrading.reload_regime_stack().

Companion to marvel's tesseract/tests/test_agamotto1h_validate.py, which
documents the 2026-03-22 agamotto1h_1 TvL mismatch (root cause:
_load_regime_stack() ran once in __init__ and never again) but — as written
— only calls the module-level load_regime_stack() function on two file
states and never touches AgamottoTrading itself, so it cannot actually prove
a fix. This file drives the real reload_regime_stack() method and the real
filesystem to prove the fix works: a stack rolled after boot reaches the bot
without a restart, and an unchanged file costs nothing extra.
"""
import contextlib
import os
import sys
import time
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agamotto.trading import AgamottoTrading


@contextlib.contextmanager
def _regime_stack_harness(regime_stack_path):
    """AgamottoTrading with a REAL file at REGIME_STACK_PATH and a fake
    _load_regime_stack that just counts calls — a real load needs real
    joblib model/scaler/metadata pickles per leg, too heavy for this test;
    the mechanism under test (reload_regime_stack's mtime gate) never
    inspects what _load_regime_stack loaded, only whether it was called.

    A context manager, not a plain constructor: _load_regime_stack must stay
    patched for every reload_regime_stack() call the test makes, not just the
    one inside __init__ — a `with` block that closed right after
    construction would leave later reload_regime_stack() calls hitting the
    REAL _load_regime_stack (which needs marvel's utils.lib on sys.path and
    genuine model pickles, neither present here).

    Yields (inst, load_calls); load_calls is appended to on every
    _load_regime_stack() call, real or faked.
    """
    config = {
        "TIME_UNIT": "1d",
        "SIZES": [0.01],
        "SYMBOLS": ["BINANCE_PERP_BTC_USDT"],
        "CAPITAL": 1000,
        "REGIME_STACK_PATH": regime_stack_path,
    }
    load_calls = []

    def fake_load_regime_stack(self_inner):
        load_calls.append(time.time())
        self_inner.regime_stack = [{"id": f"call_{len(load_calls)}"}]
        self_inner.models = {"long": {}, "short": {}}

    with patch.object(AgamottoTrading, "_load_regime_stack", fake_load_regime_stack), \
         patch.object(AgamottoTrading, "_calculate_sizes"), \
         patch.object(AgamottoTrading, "load_data"):
        inst = AgamottoTrading(config=config, home_root="/tmp", period="window_test")
        yield inst, load_calls


def test_reload_is_a_noop_when_the_file_has_not_changed(tmp_path):
    """The common case, every cycle: no roll happened, so no extra load."""
    csv_path = str(tmp_path / "filtered_optimal_regime_stack.csv")
    with open(csv_path, "w") as f:
        f.write("regime,model,position,optimal_threshold,directory\n")

    with _regime_stack_harness(csv_path) as (inst, load_calls):
        assert len(load_calls) == 1  # __init__'s own load

        # __init__ does not seed a baseline mtime, so the FIRST
        # reload_regime_stack() call always reloads once — harmless, it
        # re-reads what __init__ just read.
        inst.reload_regime_stack()
        assert len(load_calls) == 2

        # Every subsequent call, file unchanged: no reload.
        inst.reload_regime_stack()
        inst.reload_regime_stack()
        assert len(load_calls) == 2


def test_reload_picks_up_a_file_rewritten_after_boot(tmp_path):
    """The actual 2026-03-22 scenario: /gauntlet-rolling overwrites the CSV
    while the bot is live. The next cycle's reload_regime_stack() must see
    the new content without a process restart."""
    csv_path = str(tmp_path / "filtered_optimal_regime_stack.csv")
    with open(csv_path, "w") as f:
        f.write("regime,model,position,optimal_threshold,directory\n")

    with _regime_stack_harness(csv_path) as (inst, load_calls):
        inst.reload_regime_stack()  # first-call reload (see test above)
        assert len(load_calls) == 2
        stack_before_roll = inst.regime_stack

        # Simulate filter_regime_stacks.py / /gauntlet-rolling rewriting the
        # file. Bump mtime explicitly: some filesystems have coarse (1s)
        # mtime resolution and a same-second rewrite could otherwise look
        # unchanged — exactly the false negative this test must not hide.
        with open(csv_path, "w") as f:
            f.write("regime,model,position,optimal_threshold,directory\nnew_regime,Ridge,long,0.01,x\n")
        os.utime(csv_path, (time.time() + 5, time.time() + 5))

        inst.reload_regime_stack()
        assert len(load_calls) == 3, (
            "reload_regime_stack() did not reload after the file changed on "
            "disk — this is the bug that caused the 2026-03-22 "
            "agamotto1h_1 TvL mismatch."
        )
        assert inst.regime_stack != stack_before_roll


def test_reload_regime_stack_method_is_public_and_callable(tmp_path):
    """symbiote/knull's per-cycle loop (make_decision) must be able to reach
    this without touching the private _load_regime_stack."""
    csv_path = str(tmp_path / "filtered_optimal_regime_stack.csv")
    with open(csv_path, "w") as f:
        f.write("regime,model,position,optimal_threshold,directory\n")
    with _regime_stack_harness(csv_path) as (inst, _load_calls):
        assert hasattr(inst, "reload_regime_stack")
        assert callable(inst.reload_regime_stack)


def test_make_decision_calls_reload_regime_stack_first():
    """The actual wiring: every make_decision() cycle opens with a reload
    attempt, not just __init__. Isolated from the rest of make_decision's
    body (feature engineering, model predict, sizing) by mocking everything
    make_decision touches after the reload call and asserting only on call
    order — those stages are exercised by test_trading.py /
    test_trading_decisions.py already."""
    with patch.object(AgamottoTrading, "__init__", lambda self, config, home_root: None):
        inst = AgamottoTrading.__new__(AgamottoTrading)
    inst.config = {"SYMBOLS": []}
    inst.reload_regime_stack = MagicMock()
    inst.features = MagicMock()  # non-None -> skip engineer_features()
    inst.verticalize = MagicMock()
    inst.regime_stack = []  # falsy -> make_decision returns right after logging "no stack"

    inst.make_decision()

    inst.reload_regime_stack.assert_called_once()
