#pragma once
// PHASE 5 — the decision rule. PRIVATE.
//
// This file turns a bar's per-regime `y_pred` into ONE decision. It is the
// last step of the port and the only one that says "trade".
//
// SHADOW IS STILL STRUCTURAL. Nothing here sends anything: the outcome crosses
// the ABI as a `Decision` and the public AgamottoStrategy logs it. There is no
// order path in this core, in the strategy, or behind any flag.
//
// ---------------------------------------------------------------------------
// THE RULE, TRANSCRIBED — NOT INVENTED
// ---------------------------------------------------------------------------
// Three reference files, read line by line:
//
// (1) marvel `gauntlet/thresholds.py::signed_threshold` — where the SIGN lives:
//
//       long           -> C+T  (fires on y_pred > C+T)
//       short (regr.)  -> C-T  (fires on y_pred < C-T, i.e. SELECTIVE)
//
//     and `ABS_THRESH_FLOOR = 0.0002`, the 2 bps floor on the WIDTH.
//
// (2) dc `agamotto_pkg/src/agamotto/trading.py::dual_gate_filter` (lines 42-49)
//     — where a regime row fires:
//
//       long_base  = (position == "long")  & (prediction > opt_threshold)
//       short_base = (position == "short") & (prediction < opt_threshold)
//
//     `opt_threshold` is the stack row's `optimal_threshold`, which is the
//     per-leg SIGNED EDGE — the centre rides INSIDE it (thresholds.py
//     `LegGate.edge`). VERIFIED on the deployed stack: all 39 long rows carry
//     0.0005764100000000001 = THRESHOLD_CENTER_LONG (-0.00012442) +
//     THRESHOLD_LONG (0.00070083), and all 23 short rows carry -0.00104534 =
//     THRESHOLD_CENTER_SHORT (0.00030606) - THRESHOLD_SHORT (0.0013514). This
//     core computes the edge from algo_params instead of reading the stack
//     column, so the two are independent expressions of the same number and
//     `tests/decision_parity.py` grades one against the other.
//
//     The 2-BAR ARM IS NOT REPRODUCED, and that is not a gap: `dual_gate_filter`
//     applies it only when BOTH `opt_threshold_2bar` and `prediction_2bar` are
//     columns of the frame. The deployed `filtered_optimal_regime_stack.csv` has
//     twelve columns and neither is among them, so `has_dual` is False and the
//     reference takes the 1-bar branch on every bar. A core that implemented the
//     dual arm would be implementing a branch the deployed arm cannot reach.
//
// (3) dc `trading.py::make_decision` (lines 834-864) — where the VOTE happens:
//
//       reverse    = int(self.config.get("REVERSE", 1))
//       net_count  = long_count - short_count
//       final_qty  = base_size * net_count * reverse
//       self.decisions[sym] = [price, final_qty]
//
//     and knull `orb_bridge.py::_decisions_to_signals` (lines 155-169), which
//     is how that qty becomes a side:
//
//       if abs(target_qty) < 1e-9 or price <= 0: FLAT
//       side = "LONG" if target_qty > 0 else "SHORT"
//
//     `base_size` is a POSITIVE quantity (CAPITAL/price), so the sign of
//     `final_qty` is the sign of `net_count * reverse` and nothing else. That
//     is the whole side rule.
//
// THERE IS NO MIN_SIGNAL_COUNT AND NO HOLD-TTL ON THIS ARM. Both were looked
// for and neither exists:
//   * `MIN_SIGNAL_COUNT` is a MJOLNIR key (`mjolnir/README.md` line 130, and
//     every occurrence in the repo is a mjolnir test fixture). It is absent
//     from `pred_agamotto.base.15m_1/setting.json` and no agamotto or orb code
//     path reads it. The agamotto vote threshold is `net_count != 0`, i.e. an
//     effective minimum of ONE net vote.
//   * There is no signal-hold / TTL anywhere in the agamotto path. The only
//     TTL on this arm is `SIGNAL_TTL_SEC: 300.0`, which lives inside
//     `EXECUTORS[venue]` and is an ORDER lifetime in the executor — it never
//     reaches a decision. Every 15m bar re-decides from scratch and the
//     decision is a TARGET position, not an event.
// A core that added either would be quieter than the reference and would look
// exactly like one whose regimes stopped firing.
//
// ---------------------------------------------------------------------------
// THE FLOOR IS ENFORCED HERE, AND ITS VALUE IS GATED, NOT RE-DECLARED
// ---------------------------------------------------------------------------
// CLAUDE.md: `|threshold| >= 2 bps`, never selected or deployed below it. A C++
// translation unit cannot `import gauntlet.thresholds`, so the constant is
// necessarily spelled once in `decision_rule.cpp` — and `kAbsThreshFloor` is
// EXPORTED so `tests/decision_parity.py` can assert it equals the imported
// `gauntlet.thresholds.ABS_THRESH_FLOOR`. That makes it a gated copy of one
// source rather than a second independent declaration; if marvel ever moves the
// floor, the gate goes red instead of the two silently disagreeing.
//
// A sub-floor width is REFUSED AT LOAD (`GateParams::validate`, called from
// `createCore`), never clamped up to the floor. Clamping would be the banned
// `max()`-on-a-derived-quantity shape: an operator who typed 2e-5 would get a
// running strategy gated at 2e-4 and no way to tell.
//
// ---------------------------------------------------------------------------
// WHAT IS **NOT** IN THE REFERENCE AND IS THEREFORE A REPORTING CHOICE
// ---------------------------------------------------------------------------
// The reference produces a VOTE COUNT, not a `y_pred`. There is no single
// prediction behind an agamotto decision — `make_decision` never keeps one.
// `Decision::y_pred` / `threshold` / `threshold_center` / `winning_regime_code`
// therefore describe a REPRESENTATIVE regime, chosen so the reported triple is
// self-consistent (the reported y_pred really does clear the reported
// centre+/-width). The choice is documented on `evaluateDecision` and it can
// NEVER change `fired` or `side`, which are the only two fields anything could
// act on. Phase 4's "largest |y_pred| over everything that predicted" ordering
// is deliberately NOT built on: it could name a SHORT regime as the winner of a
// LONG decision.
#include <cstdint>
#include <string>
#include <vector>

#include "regime_gate.hpp"   // Position

namespace agamotto {

// CLAUDE.md's 2 bps floor on the gate HALF-WIDTH. Exported so the parity gate
// can grade it against `gauntlet.thresholds.ABS_THRESH_FLOOR` rather than this
// being a second, ungraded declaration of the same number.
extern const double kAbsThreshFloor;

// Float-comparison slack, mirroring `thresholds.py::_FLOOR_EPS`: a nominal
// 0.0002 round-trips through JSON a few ULPs low and a value that IS the floor
// must not be rejected.
extern const double kFloorEps;

// One leg's gate. `edge` is what `y_pred` is actually compared against, and it
// MAY BE NEGATIVE ON A LONG LEG (it is -0.000689 on one measured arm and
// +0.00057641 on the deployed one). Nothing may infer a position from its sign.
struct LegGate {
    double width{0.0};    // the unsigned half-width T
    double center{0.0};   // the SIGNED location C
    double edge{0.0};     // C+T for long, C-T for short
};

// `gauntlet/thresholds.py::signed_threshold`, transcribed. `magnitude` is taken
// as |magnitude| exactly as the reference does (`mag = abs(float(magnitude))`).
//
// The CLASSIFICATION arm of the reference (`return mag`, ignoring the centre)
// is deliberately absent: it applies to `USE_CLASSIFICATION` experiments whose
// y_pred is a probability, agamotto is a regression arm, and a branch that
// cannot be reached from this core's configuration would be untested code that
// silently ignores the centre.
double signedThreshold(double magnitude, Position pos, double center);

// The gate as it arrives from algo_params. Field names are IDENTICAL to
// `agamotto::DecisionGate` in the public contract header and to the
// `threshold_long` / `threshold_short` / `threshold_center_long` /
// `threshold_center_short` / `reverse` keys `make_sentinel_config.py` emits, so
// the copy at each seam is name-for-name and a transposition is visible by
// reading one line against another.
struct GateParams {
    double threshold_long{0.0};
    double threshold_short{0.0};
    double threshold_center_long{0.0};
    double threshold_center_short{0.0};
    int    reverse{0};

    // Throws std::invalid_argument on anything that cannot be a live gate:
    //   * a non-finite width or centre;
    //   * a NEGATIVE width (the sign belongs to the position, not the config —
    //     `thresholds.py::validate_threshold` raises on the same thing);
    //   * a width below `kAbsThreshFloor` (CLAUDE.md hard rule). NOT clamped;
    //   * `reverse` outside {-1, +1}.
    //
    // On `reverse`: the reference multiplies a QUANTITY by it
    // (`final_qty = base_size * net_count * reverse`) and never validates it,
    // so REVERSE=0 silently produces a permanently flat bot and REVERSE=2
    // silently doubles live size — the bridge recovers `net_count` back out of
    // the qty. A shadow core reports a SIDE and cannot represent either, so
    // both are refused here rather than being mapped onto a plausible side.
    // Reported as a reference weakness, not fixed there.
    void validate() const;

    LegGate leg(Position pos) const;
};

// `dual_gate_filter`'s 1-bar arm for ONE row: long fires above its edge, short
// fires BELOW its edge. A non-finite `y_pred` never fires — `inf > edge` is
// true, and a regime that fires off a poisoned column is worse than one that
// does not fire (the same exclusion `runModels` already applies).
bool legFires(double y_pred, const LegGate& gate, Position pos);

// What one bar's decision is, before it is copied into the ABI's `Decision`.
struct DecisionOutcome {
    bool   fired{false};
    int    side{0};                 // +1 long, -1 short, 0 flat
    double y_pred{0.0};             // the REPRESENTATIVE regime's (see below)
    double threshold{0.0};          // that regime's leg WIDTH
    double threshold_center{0.0};   // that regime's leg CENTRE
    int    n_triggered{0};          // votes cast: long_votes + short_votes
    int    n_long{0};               // `long_count`  in make_decision
    int    n_short{0};              // `short_count` in make_decision
    int    net_count{0};            // long_count - short_count, BEFORE reverse
    int    winning_index{-1};       // stack index of the representative regime
};

// One bar's decision from the per-regime predictions.
//
// `positions[i]` is stack regime i's leg. `y_pred[i]` is its prediction, or NaN
// when the REGIME GATE did not let that bar through (the reference never
// predicts a filtered-out row at all — `predict(filtered_signals)`), or when
// the model produced a non-finite number. Both are "no vote", and the
// distinction is already counted upstream by `nonfinite_predictions`.
//
// THE VOTE (make_decision:844-861, verbatim):
//   n_long  = #{i : positions[i] == LONG  and y_pred[i] >  C_long  + T_long }
//   n_short = #{i : positions[i] == SHORT and y_pred[i] <  C_short - T_short}
//   net     = n_long - n_short
//   side    = sign(net * reverse)          [orb_bridge:168, |qty|<1e-9 -> FLAT]
//   fired   = side != 0
//
// THE REPRESENTATIVE REGIME (a REPORTING choice, see the file banner):
// among the regimes that VOTED ON THE MAJORITY LEG — the leg whose count
// decided `net`, taken BEFORE `reverse`, because that is the leg whose
// predictions produced the number — the one with the largest |y_pred|; ties go
// to the lowest stack index. When `net == 0` (a tie, or nothing voted) there is
// no majority leg, so the pick falls back to all voters; when nothing voted at
// all, `winning_index` is -1 and y_pred/threshold/threshold_center are 0.
//
// Throws std::invalid_argument when the two vectors disagree in length: a
// prediction paired with the wrong regime's leg would flip votes silently.
DecisionOutcome evaluateDecision(const GateParams& gate,
                                 const std::vector<Position>& positions,
                                 const std::vector<double>& y_pred);

} // namespace agamotto
