// PHASE 3 — the regime predicates. See regime_gate.hpp for the contract, the
// codes-only rule, and the reference quirks this file reproduces deliberately.
//
// EVERY predicate below is a transcription of one branch of
// agamotto_pkg/src/agamotto/research_filters.py `apply_filter_mask`. The
// reference line is quoted next to each so a change there is diffable against
// this file rather than requiring the whole function to be re-derived.
//
// The comparison operators are the thing being ported. `>` vs `>=` on
// `adx > 25` moves a boolean on every bar that lands exactly on 25, and the
// only way that is caught is tests/regime_parity.py's EXACT mask equality
// against research_filters itself — no tolerance, because a mask is a decision,
// not a measurement.
#include "regime_gate.hpp"
#include "codes_generated.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace agamotto {
namespace {

// NaN COMPARES FALSE, in every predicate, exactly as pandas does. This is not
// defensive coding — it is the mechanism that keeps the 53 r07x-gated regimes
// inert (the q80/q90/q95 cutoffs are all-NaN on a 699-row panel), so a
// "helpful" NaN-to-zero anywhere in this file would turn 53 non-firing regimes
// into 53 regimes comparing against 0.0. C++'s `<` and `>` already yield false
// for NaN operands; the helpers below rely on that and add nothing.
template <typename F>
std::vector<char> cmp1(const std::vector<double>& x, F pred)
{
    std::vector<char> m(x.size(), 0);
    for (size_t i = 0; i < x.size(); ++i) m[i] = pred(x[i]) ? 1 : 0;
    return m;
}

template <typename F>
std::vector<char> cmp2(const std::vector<double>& a, const std::vector<double>& b, F pred)
{
    if (a.size() != b.size()) {
        throw std::invalid_argument("regime_gate: ragged panel columns");
    }
    std::vector<char> m(a.size(), 0);
    for (size_t i = 0; i < a.size(); ++i) m[i] = pred(a[i], b[i]) ? 1 : 0;
    return m;
}

void andInto(std::vector<char>& acc, const std::vector<char>& rhs)
{
    if (acc.size() != rhs.size()) {
        throw std::invalid_argument("regime_gate: ragged masks");
    }
    for (size_t i = 0; i < acc.size(); ++i) acc[i] = (acc[i] && rhs[i]) ? 1 : 0;
}

// research_filters `_volume_ratio`: quote_vol_ratio FIRST, vol_ratio second,
// raise if neither. The order is not cosmetic — both columns exist in the live
// panel and they are different numbers, so reading the wrong one silently
// changes which bars are "high volume".
const std::vector<double>& volumeRatio(const Table& t)
{
    if (t.has(codes::F_QUOTE_VOL_RATIO)) return t.get(codes::F_QUOTE_VOL_RATIO);
    if (t.has(codes::F_VOL_RATIO)) return t.get(codes::F_VOL_RATIO);
    // research_filters raises here rather than substituting a scalar 1.0, which
    // would return a BOOL instead of a per-row mask. Same discipline.
    throw std::invalid_argument(
        "regime_gate: the panel carries neither volume-ratio column, so a "
        "volume regime cannot be evaluated. A constant fallback here fires on "
        "every bar.");
}

// The `up_trend` / `down_trend` pair the MVG-dependent block is built on
// (research_filters: `up_trend = (close > mvg1) & (mvg1 > mvg2)`).
// mvg1/mvg2/mvg3 and close carry their REAL names: dc/obfuscation/map.json has
// no code for any of the four (see agamotto_core/README.md, "The mvg gap").
// They are not distinctive regime names and the audit does not gate on them.
std::vector<char> trendMask(const Table& t, Position pos)
{
    const std::vector<double>& c = t.get("close");
    const std::vector<double>& m1 = t.get("mvg1");
    const std::vector<double>& m2 = t.get("mvg2");
    if (pos == Position::LONG) {
        std::vector<char> up = cmp2(c, m1, [](double a, double b) { return a > b; });
        andInto(up, cmp2(m1, m2, [](double a, double b) { return a > b; }));
        return up;
    }
    std::vector<char> down = cmp2(c, m1, [](double a, double b) { return a < b; });
    andInto(down, cmp2(m1, m2, [](double a, double b) { return a < b; }));
    return down;
}

} // namespace

bool atomIsKnown(uint16_t code)
{
    switch (code) {
        case codes::R_HIGH_VOL_Q80:
        case codes::R_HIGH_VOL_Q90:
        case codes::R_HIGH_VOL_Q95:
        case codes::R_LOW_VOL:
        case codes::R_HIGH_VOL:
        case codes::R_LOW_VOLUME:
        case codes::R_HIGH_VOLUME:
        case codes::R_VOL_BREAKOUT:
        case codes::R_STRONG_CANDLE:
        case codes::R_CCI_REVERSAL:
        case codes::R_MOM_POSITIVE:
        case codes::R_BUY_PRESSURE:
        case codes::R_RSI_OVERSOLD:
        case codes::R_RSI_OVERBOUGHT:
        case codes::R_MACD_BULLISH:
        case codes::R_MACD_BEARISH:
        case codes::R_MFI_OVERSOLD:
        case codes::R_MFI_OVERBOUGHT:
        case codes::R_BOP_BULLISH:
        case codes::R_BOP_BEARISH:
        case codes::R_ROC_POSITIVE:
        case codes::R_ROC_NEGATIVE:
        case codes::R_STOCH_BULLISH:
        case codes::R_ADX_TREND:
        case codes::R_STRONG_TREND:
        case codes::R_MA_MOMENTUM:
        case codes::R_ABOVE_ALL_MAS:
        case codes::R_NEAR_MA:
        case codes::R_BB_REBOUND:
        case codes::R_SAR_ALIGNED:
            return true;
        default:
            // Everything else in codes_generated.hpp is a regime from ANOTHER
            // algo's inventory (tick book/OI/funding/liquidation states,
            // valkyrie's option-surface states). They have codes because
            // map.json is shared; they have no columns in a 15m kline panel and
            // no branch in research_filters' kline path, so they are unknown
            // HERE and saying so is the accurate answer.
            return false;
    }
}

bool positionAllowed(const std::vector<uint16_t>& atom_codes, Position pos)
{
    // research_filters LONG_ONLY_FILTERS / SHORT_ONLY_FILTERS, applied per
    // ATOM. The reference evaluates each conjunct through apply_filter_mask
    // separately, so a conjunction is refused for a position exactly when ANY
    // of its atoms refuses it — and a conjunction carrying one long-only and
    // one short-only atom is refused on BOTH sides (the reference's `[]`).
    bool needs_long = false;
    bool needs_short = false;
    for (const uint16_t c : atom_codes) {
        switch (c) {
            case codes::R_MACD_BULLISH:   // research_filters LONG_ONLY_FILTERS
            case codes::R_STOCH_BULLISH:
            case codes::R_RSI_OVERSOLD:
            case codes::R_MFI_OVERSOLD:
            case codes::R_BOP_BULLISH:
            case codes::R_ROC_POSITIVE:
                needs_long = true;
                break;
            case codes::R_MACD_BEARISH:   // research_filters SHORT_ONLY_FILTERS
            case codes::R_RSI_OVERBOUGHT:
            case codes::R_MFI_OVERBOUGHT:
            case codes::R_BOP_BEARISH:
            case codes::R_ROC_NEGATIVE:
                needs_short = true;
                break;
            default:
                break;
        }
    }
    if (needs_long && needs_short) return false;   // reference returns []
    if (needs_long) return pos == Position::LONG;
    if (needs_short) return pos == Position::SHORT;
    return true;
}

std::vector<char> atomMask(const Table& panel, uint16_t code, Position pos)
{
    const bool lng = (pos == Position::LONG);

    // --- the position gate, FIRST -----------------------------------------
    // research_filters checks `allowed_positions` before it evaluates
    // anything, so a long-only atom asked for a short mask is all-False and
    // its short branch is never reached. Reproduced here at the same point,
    // which is what makes the reference's unreachable short `stoch_bullish`
    // branch unreachable in this port too.
    if (!positionAllowed({code}, pos)) {
        return std::vector<char>(panel.cols.empty() ? 0 : panel.cols.front().size(), 0);
    }

    switch (code) {
        // --- the trailing vol-quantile ENTRY gates -------------------------
        // Resolved ABOVE the position split in the reference, because the
        // predicate is position-INVARIANT: it says nothing about direction,
        // only whether the bar's range cleared a trailing cutoff.
        //
        // *** THESE THREE ARE WHY 53 OF 62 DEPLOYED REGIMES ARE INERT LIVE. ***
        // The cutoff columns are rolling(700, min_periods=700) on a 699-row
        // panel, i.e. NaN on every row, and `x > NaN` is False. See the banner
        // in regime_gate.hpp and marvel PR #532 /
        // docs/findings/2026-08-19-vol-quantile-regimes-inert-live.md. There is
        // deliberately NO rolling fallback here: research_filters has none
        // either, and one would compute a cutoff across symbol boundaries on
        // the stacked vertical panel.
        case codes::R_HIGH_VOL_Q80:
            return cmp2(panel.get(codes::F_PRICE_RANGE_PCT),
                        panel.get(codes::F_PRICE_RANGE_PCT_Q80),
                        [](double a, double b) { return a > b; });
        case codes::R_HIGH_VOL_Q90:
            return cmp2(panel.get(codes::F_PRICE_RANGE_PCT),
                        panel.get(codes::F_PRICE_RANGE_PCT_Q90),
                        [](double a, double b) { return a > b; });
        case codes::R_HIGH_VOL_Q95:
            return cmp2(panel.get(codes::F_PRICE_RANGE_PCT),
                        panel.get(codes::F_PRICE_RANGE_PCT_Q95),
                        [](double a, double b) { return a > b; });

        // --- the trailing-MEDIAN split -------------------------------------
        // Duplicated verbatim into both position branches in the reference,
        // with the SAME predicate on each side. q50 is rolling(700,
        // min_periods=1), so unlike q80/q90/q95 it is populated on every row.
        case codes::R_LOW_VOL:
            return cmp2(panel.get(codes::F_PRICE_RANGE_PCT),
                        panel.get(codes::F_PRICE_RANGE_PCT_Q50),
                        [](double a, double b) { return a < b; });
        case codes::R_HIGH_VOL:
            return cmp2(panel.get(codes::F_PRICE_RANGE_PCT),
                        panel.get(codes::F_PRICE_RANGE_PCT_Q50),
                        [](double a, double b) { return a > b; });

        // --- volume ratios --------------------------------------------------
        // Identical on both sides in the reference. The thresholds are bare
        // literals there and are bare literals here: 1.0 / 1.0 / 2.0.
        case codes::R_LOW_VOLUME:
            return cmp1(volumeRatio(panel), [](double v) { return v < 1.0; });
        case codes::R_HIGH_VOLUME:
            return cmp1(volumeRatio(panel), [](double v) { return v > 1.0; });
        case codes::R_VOL_BREAKOUT:
            return cmp1(volumeRatio(panel), [](double v) { return v > 2.0; });

        // --- sign-flipped pairs ---------------------------------------------
        case codes::R_STRONG_CANDLE:
            // long: open_close_pct > 0.005; short: < -0.005. NOT |x| > 0.005.
            return lng ? cmp1(panel.get(codes::F_OPEN_CLOSE_PCT),
                              [](double v) { return v > 0.005; })
                       : cmp1(panel.get(codes::F_OPEN_CLOSE_PCT),
                              [](double v) { return v < -0.005; });
        case codes::R_CCI_REVERSAL:
            return lng ? cmp1(panel.get(codes::F_CCI), [](double v) { return v > 100.0; })
                       : cmp1(panel.get(codes::F_CCI), [](double v) { return v < -100.0; });
        case codes::R_MOM_POSITIVE:
            // THE NAME LIES ON THE SHORT SIDE: the reference's short branch is
            // `mom < 0`. It is not in SHORT_ONLY_FILTERS, so short is allowed
            // and the predicate is inverted rather than the regime refused.
            return lng ? cmp1(panel.get(codes::F_MOM), [](double v) { return v > 0.0; })
                       : cmp1(panel.get(codes::F_MOM), [](double v) { return v < 0.0; });
        case codes::R_BUY_PRESSURE:
            // Asymmetric on purpose: 0.55 / 0.45, not one threshold mirrored.
            return lng ? cmp1(panel.get(codes::F_BUY_PRESSURE),
                              [](double v) { return v > 0.55; })
                       : cmp1(panel.get(codes::F_BUY_PRESSURE),
                              [](double v) { return v < 0.45; });

        // --- direction-locked atoms (allowed_positions already filtered) ----
        case codes::R_RSI_OVERSOLD:      // long only
            return cmp1(panel.get(codes::F_RSI), [](double v) { return v < 30.0; });
        case codes::R_RSI_OVERBOUGHT:    // short only
            return cmp1(panel.get(codes::F_RSI), [](double v) { return v > 70.0; });
        case codes::R_MACD_BULLISH:      // long only
            return cmp1(panel.get(codes::F_MACDHIST), [](double v) { return v > 0.0; });
        case codes::R_MACD_BEARISH:      // short only
            return cmp1(panel.get(codes::F_MACDHIST), [](double v) { return v < 0.0; });
        case codes::R_MFI_OVERSOLD:      // long only
            return cmp1(panel.get(codes::F_MFI), [](double v) { return v < 30.0; });
        case codes::R_MFI_OVERBOUGHT:    // short only
            return cmp1(panel.get(codes::F_MFI), [](double v) { return v > 70.0; });
        case codes::R_BOP_BULLISH:       // long only
            return cmp1(panel.get(codes::F_BOP), [](double v) { return v > 0.1; });
        case codes::R_BOP_BEARISH:       // short only
            return cmp1(panel.get(codes::F_BOP), [](double v) { return v < -0.1; });
        case codes::R_ROC_POSITIVE:      // long only
            return cmp1(panel.get(codes::F_ROC), [](double v) { return v > 0.0; });
        case codes::R_ROC_NEGATIVE:      // short only
            return cmp1(panel.get(codes::F_ROC), [](double v) { return v < 0.0; });
        case codes::R_STOCH_BULLISH:
            // LONG ONLY. The reference ALSO carries `stoch_k > stoch_d` — the
            // same predicate, not the mirror — in its short branch, which
            // allowed_positions makes unreachable. Ported as unreachable
            // rather than "corrected" to `<`, which would make it fire.
            return cmp2(panel.get(codes::F_STOCH_K), panel.get(codes::F_STOCH_D),
                        [](double a, double b) { return a > b; });

        // --- position-invariant --------------------------------------------
        case codes::R_ADX_TREND:
            // Identical on both sides: `adx > 25`, strictly greater.
            return cmp1(panel.get(codes::F_ADX), [](double v) { return v > 25.0; });

        // --- MVG_DEPENDENT_FILTERS ------------------------------------------
        // The reference reaches these only after both position branches fall
        // through, and requires close/mvg1/mvg2 to exist (raising otherwise).
        // Table::get throws on a missing column, which is the same discipline.
        case codes::R_STRONG_TREND: {
            std::vector<char> m = trendMask(panel, pos);
            const std::vector<double>& m2 = panel.get("mvg2");
            const std::vector<double>& m3 = panel.get("mvg3");
            andInto(m, lng ? cmp2(m2, m3, [](double a, double b) { return a > b; })
                           : cmp2(m2, m3, [](double a, double b) { return a < b; }));
            return m;
        }
        case codes::R_MA_MOMENTUM: {
            // NOTE: no close term at all — this is purely the MA stack's own
            // ordering, unlike strong_trend which is gated on up/down_trend.
            const std::vector<double>& m1 = panel.get("mvg1");
            const std::vector<double>& m2 = panel.get("mvg2");
            const std::vector<double>& m3 = panel.get("mvg3");
            std::vector<char> m = lng ? cmp2(m1, m2, [](double a, double b) { return a > b; })
                                      : cmp2(m1, m2, [](double a, double b) { return a < b; });
            andInto(m, lng ? cmp2(m1, m3, [](double a, double b) { return a > b; })
                           : cmp2(m1, m3, [](double a, double b) { return a < b; }));
            return m;
        }
        case codes::R_ABOVE_ALL_MAS: {
            // The name is long-shaped; the SHORT branch is "below all MAs".
            const std::vector<double>& c = panel.get("close");
            std::vector<char> m = lng
                ? cmp2(c, panel.get("mvg1"), [](double a, double b) { return a > b; })
                : cmp2(c, panel.get("mvg1"), [](double a, double b) { return a < b; });
            andInto(m, lng
                ? cmp2(c, panel.get("mvg2"), [](double a, double b) { return a > b; })
                : cmp2(c, panel.get("mvg2"), [](double a, double b) { return a < b; }));
            andInto(m, lng
                ? cmp2(c, panel.get("mvg3"), [](double a, double b) { return a > b; })
                : cmp2(c, panel.get("mvg3"), [](double a, double b) { return a < b; }));
            return m;
        }
        case codes::R_NEAR_MA: {
            // A `<` on a SIGNED relative distance, NOT a |distance| band: on
            // the long side every bar BELOW mvg1 gives a negative distance and
            // satisfies it. Written as the reference writes it — the division
            // by mvg1, then the comparison — so a zero mvg1 yields the same
            // +/-inf or NaN pandas would produce and the same boolean.
            const std::vector<double>& c = panel.get("close");
            const std::vector<double>& m1 = panel.get("mvg1");
            return lng ? cmp2(c, m1, [](double a, double b) { return (a - b) / b < 0.02; })
                       : cmp2(c, m1, [](double a, double b) { return (b - a) / b < 0.02; });
        }
        case codes::R_BB_REBOUND:
            // long: close BELOW the lower band; short: close ABOVE the upper.
            // The two read DIFFERENT columns, so a mirrored implementation on
            // one column would be wrong on one side and plausible on both.
            return lng ? cmp2(panel.get("close"), panel.get(codes::F_BB_LOWER),
                              [](double a, double b) { return a < b; })
                       : cmp2(panel.get("close"), panel.get(codes::F_BB_UPPER),
                              [](double a, double b) { return a > b; });
        case codes::R_SAR_ALIGNED:
            return lng ? cmp2(panel.get("close"), panel.get(codes::F_SAR),
                              [](double a, double b) { return a > b; })
                       : cmp2(panel.get("close"), panel.get(codes::F_SAR),
                              [](double a, double b) { return a < b; });

        default:
            break;
    }

    // The reference's `strict_filters=True` path raises here; its
    // strict_filters=False path returns an all-False mask. This ALWAYS throws.
    // A live core that quietly gated an unknown regime to "never fires" is a
    // strategy that is switched off with nothing in the log saying so — and an
    // all-TRUE fallback would be worse still, an unconditional `baseline`.
    throw std::invalid_argument("regime_gate: unknown regime atom code " +
                                std::to_string(static_cast<unsigned>(code)));
}

std::vector<char> regimeMask(const Table& panel,
                             const std::vector<uint16_t>& atom_codes,
                             Position pos)
{
    if (atom_codes.empty()) {
        // The reference's `mask if mask is not None else Series(True)` would
        // give an all-True mask for an empty conjunction. That is `baseline`,
        // the unconditional fire-on-every-bar regime CLAUDE.md removed forever
        // (2026-06-18). Refused rather than reproduced.
        throw std::invalid_argument(
            "regime_gate: an EMPTY regime is the unconditional always-fire "
            "gate; refusing to evaluate it");
    }
    if (atom_codes.size() > MAX_REGIME_ATOMS) {
        throw std::invalid_argument(
            "regime_gate: " + std::to_string(atom_codes.size()) + " atoms "
            "exceeds MAX_REGIME_ATOMS; truncating would LOOSEN the gate");
    }
    const size_t n = panel.cols.empty() ? 0 : panel.cols.front().size();

    // The position gate is applied to the CONJUNCTION as a whole here, which
    // matches the reference: it evaluates each conjunct through
    // apply_filter_mask, and any conjunct that refuses the position returns
    // all-False, which ANDs the whole thing to all-False.
    if (!positionAllowed(atom_codes, pos)) {
        return std::vector<char>(n, 0);
    }

    std::vector<char> mask = atomMask(panel, atom_codes[0], pos);
    for (size_t i = 1; i < atom_codes.size(); ++i) {
        andInto(mask, atomMask(panel, atom_codes[i], pos));
    }
    return mask;
}

} // namespace agamotto
