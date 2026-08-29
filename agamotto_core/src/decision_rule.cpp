// PHASE 5 — the decision rule. See decision_rule.hpp for the transcription and
// for every reference line number this file reproduces.
#include "decision_rule.hpp"

#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace agamotto {

// CLAUDE.md's 2 bps floor. THE value lives in marvel
// `gauntlet/thresholds.py::ABS_THRESH_FLOOR`; this is a gated copy —
// `tests/decision_parity.py` imports that module and asserts the two agree, so
// a change on the marvel side turns the gate red instead of leaving two
// independent numbers to drift.
const double kAbsThreshFloor = 0.0002;

// `thresholds.py::_FLOOR_EPS`.
const double kFloorEps = 1e-12;

namespace {

std::string fmt(double v)
{
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.17g", v);
    return std::string(buf);
}

void requireFinite(double v, const char* what)
{
    if (!std::isfinite(v)) {
        throw std::invalid_argument(
            std::string("agamotto decision gate: ") + what + "=" + fmt(v) +
            " is not finite. The gate is compared against every y_pred; a "
            "non-finite bound gates every bar the same way and says nothing.");
    }
}

// `thresholds.py::validate_threshold`, minus the string-to-float step.
void validateWidth(double width, const char* key)
{
    requireFinite(width, key);
    if (width < 0.0) {
        throw std::invalid_argument(
            std::string("agamotto decision gate: ") + key + "=" + fmt(width) +
            " is negative. Store the unsigned MAGNITUDE; the sign is applied "
            "per position by signedThreshold() (long C+T, short C-T).");
    }
    if (width < kAbsThreshFloor - kFloorEps) {
        throw std::invalid_argument(
            std::string("agamotto decision gate: ") + key + "=" + fmt(width) +
            " is below the " + fmt(kAbsThreshFloor) + " floor (|threshold| >= "
            "2 bps, CLAUDE.md hard rule). A sub-floor threshold fires on "
            "essentially every bar, which is the always-on failure mode the "
            "banned `baseline` regime had. REFUSED, not raised to the floor: "
            "clamping it would leave a strategy running at a gate nobody chose "
            "and nothing reporting the difference.");
    }
}

} // namespace

double signedThreshold(double magnitude, Position pos, double center)
{
    // `mag = abs(float(magnitude))` — the reference takes the magnitude even
    // when handed a signed number, and validate() has already refused a
    // negative one on the configured path.
    const double mag = std::fabs(magnitude);
    return (pos == Position::LONG) ? (center + mag) : (center - mag);
}

void GateParams::validate() const
{
    validateWidth(threshold_long, "threshold_long");
    validateWidth(threshold_short, "threshold_short");
    // A LOCATION, not a half-width: it may be negative (it IS negative on the
    // deployed long leg) and it has no floor. Only the width controls
    // selectivity — `thresholds.py::read_threshold_center`.
    requireFinite(threshold_center_long, "threshold_center_long");
    requireFinite(threshold_center_short, "threshold_center_short");

    if (reverse != 1 && reverse != -1) {
        throw std::invalid_argument(
            "agamotto decision gate: reverse=" + std::to_string(reverse) +
            " must be +1 or -1. The reference multiplies a QUANTITY by this "
            "(trading.py:861 final_qty = base_size * net_count * reverse) and "
            "never validates it, so 0 is a permanently flat bot and 2 is a "
            "silent doubling of live size. A shadow core reports a SIDE and "
            "can represent neither, so both are refused here rather than "
            "mapped onto a plausible side.");
    }
}

LegGate GateParams::leg(Position pos) const
{
    LegGate g;
    if (pos == Position::LONG) {
        g.width = threshold_long;
        g.center = threshold_center_long;
    } else {
        g.width = threshold_short;
        g.center = threshold_center_short;
    }
    g.edge = signedThreshold(g.width, pos, g.center);
    return g;
}

bool legFires(double y_pred, const LegGate& gate, Position pos)
{
    // `inf > edge` is true and `-inf < edge` is true, so a non-finite
    // prediction would fire on one leg or the other. Excluded here for the same
    // reason runModels() excludes it from n_triggered: a regime that votes off
    // a poisoned column is worse than one that does not vote.
    if (!std::isfinite(y_pred)) {
        return false;
    }
    // dual_gate_filter:44 / :48. STRICT inequality on both legs, and the short
    // leg is a `<` against a SIGNED edge — comparing shorts against +T is the
    // 2026-06 bug that fired on nearly every bar.
    return (pos == Position::LONG) ? (y_pred > gate.edge) : (y_pred < gate.edge);
}

DecisionOutcome evaluateDecision(const GateParams& gate,
                                 const std::vector<Position>& positions,
                                 const std::vector<double>& y_pred)
{
    if (positions.size() != y_pred.size()) {
        throw std::invalid_argument(
            "agamotto evaluateDecision: " + std::to_string(positions.size()) +
            " regime positions against " + std::to_string(y_pred.size()) +
            " predictions. A prediction paired with the wrong regime's leg "
            "flips votes with nothing downstream able to see it.");
    }

    const LegGate long_gate = gate.leg(Position::LONG);
    const LegGate short_gate = gate.leg(Position::SHORT);

    DecisionOutcome out;
    std::vector<char> voted(positions.size(), 0);

    for (size_t i = 0; i < positions.size(); ++i) {
        const LegGate& g = (positions[i] == Position::LONG) ? long_gate : short_gate;
        if (!legFires(y_pred[i], g, positions[i])) {
            continue;
        }
        voted[i] = 1;
        if (positions[i] == Position::LONG) ++out.n_long;
        else                                ++out.n_short;
    }

    out.n_triggered = out.n_long + out.n_short;
    out.net_count = out.n_long - out.n_short;

    // make_decision:861 then orb_bridge:156/168. `base_size` is CAPITAL/price
    // and therefore strictly positive, so the qty's sign is exactly this.
    const long long signed_net = static_cast<long long>(out.net_count) * gate.reverse;
    out.side = (signed_net > 0) ? +1 : ((signed_net < 0) ? -1 : 0);
    out.fired = (out.side != 0);

    // ---- the REPRESENTATIVE regime (reporting only; see the header) --------
    // The MAJORITY leg is taken from net_count BEFORE reverse: it is the leg
    // whose predictions produced the number. With reverse = -1 the reported
    // side is the opposite of the reported regime's leg, which is exactly what
    // reverse means and is why the two are kept distinct.
    const bool have_majority = (out.net_count != 0);
    const Position majority = (out.net_count > 0) ? Position::LONG : Position::SHORT;

    double best_abs = -1.0;
    for (size_t i = 0; i < positions.size(); ++i) {
        if (!voted[i]) continue;
        if (have_majority && positions[i] != majority) continue;
        const double a = std::fabs(y_pred[i]);
        if (a > best_abs) {   // strict: ties keep the LOWEST stack index
            best_abs = a;
            out.winning_index = static_cast<int>(i);
        }
    }
    if (out.winning_index >= 0) {
        const size_t w = static_cast<size_t>(out.winning_index);
        const LegGate& g = (positions[w] == Position::LONG) ? long_gate : short_gate;
        out.y_pred = y_pred[w];
        out.threshold = g.width;
        out.threshold_center = g.center;
    }
    return out;
}

} // namespace agamotto
