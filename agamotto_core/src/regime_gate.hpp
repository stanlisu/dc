#pragma once
// PHASE 3 — the regime gate. PRIVATE: the predicates ARE the alpha.
//
// Transcribed from agamotto_pkg/src/agamotto/research_filters.py
// `apply_filter_mask` / `allowed_positions`, read line by line rather than
// inferred from the regime names. Several of the predicates do NOT do what
// their name says (see "reference quirks" below) and every one of those is a
// column of wrong booleans if it is guessed.
//
// ---------------------------------------------------------------------------
// CODES ONLY, AND THAT IS AN ARTIFACT PROPERTY, NOT A STYLE.
//
// Atoms arrive as uint16 codes (`r069` -> 69) and are dispatched with a switch.
// No real regime name is a string literal anywhere in this file or in the
// column keys it reads, so `strings libagamotto_core.so` recovers none of them.
// The earlier mjolnir design (sentinel_core/src/regime_gate.cpp) takes a NAME
// and decodes it tolerantly; that was correct there while coded stacks were
// rolling out, and it is exactly what put 118 recoverable names in that
// artifact. Do not reintroduce a name table here for "convenience".
//
// Conjunctions arrive as an ARRAY of codes, not as a parsed string. mjolnir's
// single-atom `^r\d{2,}_(long|short)$` shape does not describe agamotto at all:
// 34 of the 62 deployed regimes are THREE atoms (`r029_and_r001_and_r073_long`)
// and 28 are two. Splitting is the caller's job — the strategy does it on the
// public side, where the only thing it handles is digits and separators.
// ---------------------------------------------------------------------------
//
// THE GATE IS PANEL-WIDE, NOT ROW-WISE, AND THAT IS LOAD-BEARING.
//
// Every predicate here is a per-row comparison against a column the FEATURE
// ENGINE already computed over the whole 699-bar panel — including
// `price_range_pct_q50`, which is a 700-bar rolling median. Evaluating the
// engine on a one-row frame would make that median equal the value, so
// `high_vol`/`low_vol` would be decided by a tie and the regime would silently
// never fire. The gate therefore takes the same `Table` the engine emits, and
// returns a mask of the same height; the caller reads the LAST row.
//
// ---------------------------------------------------------------------------
// *** 53 OF THE 62 DEPLOYED REGIMES CANNOT FIRE LIVE, AND THIS PORT KEEPS IT
//     THAT WAY. ***
//
// The atoms r073 / r074 / r075 compare `price_range_pct` against
// `price_range_pct_q80 / q90 / q95`, which research.py:371-376 builds as
// `rolling(VOL_Q_WINDOW=700, min_periods=700)`. The live panel is 699 rows
// (see PANEL_BARS), so min_periods is never met and all three cutoff columns
// are NaN on every row. `x > NaN` is False, so every regime carrying one of
// those atoms is INERT live — 53 of the 62 in
// `pred_agamotto.base.15m_1/filtered_optimal_regime_stack.csv`.
//
// That is TODAY'S LIVE BEHAVIOUR and it is the subject of an open production
// finding: marvel PR #532,
// docs/findings/2026-08-19-vol-quantile-regimes-inert-live.md. Reproducing it
// is the whole point — a port that "fixed" the min_periods, or widened the
// panel to 700, would start 53 regimes firing against Ridge weights that were
// never trained on a firing regime, and the port would look like the cause of
// whatever happened next. When the finding is resolved it is resolved in
// research.py FIRST and mirrored here.
//
// tests/regime_parity.py asserts BOTH halves of it explicitly — the 53 produce
// all-False masks, and the other 9 produce masks that are neither all-False nor
// all-True — because a gate that never fires would otherwise pass a
// mask-equality test trivially, on both sides, forever.
// ---------------------------------------------------------------------------
//
// REFERENCE QUIRKS REPRODUCED VERBATIM. Each of these is a place where writing
// what the name suggests gives a different column of booleans:
//
//   * `mom_positive` on the SHORT side is `mom < 0` (research_filters:~318).
//     It is not in SHORT_ONLY_FILTERS, so short is allowed and the predicate
//     is inverted rather than refused.
//   * `stoch_bullish` on the SHORT side is `stoch_k > stoch_d` — IDENTICAL to
//     the long side, not the mirror. It is DEAD CODE in the reference:
//     `stoch_bullish` is in LONG_ONLY_FILTERS, so `allowed_positions` returns
//     ["long"] and the short branch is unreachable. Ported as unreachable too,
//     rather than "corrected" to `stoch_k < stoch_d`, which would make it fire.
//   * `low_vol` / `high_vol` are duplicated verbatim into BOTH position
//     branches with the same predicate, while the `high_vol_q*` atoms are
//     resolved ONCE above the split. Same effect, different structure; the
//     structure is kept so the diff against the reference stays readable.
//   * `near_ma` is a `<` on a SIGNED distance, so on the long side every bar
//     BELOW mvg1 satisfies it (the distance is negative). It is not a
//     |distance| band.
//   * the volume ratio prefers `quote_vol_ratio` and falls back to `vol_ratio`
//     (research_filters `_volume_ratio`), in that order. Both exist in the live
//     panel, so quote_vol_ratio is what production reads; the order still
//     matters because they are different numbers.
//
// REFERENCE BEHAVIOUR NOT REPRODUCED, DECLARED:
//
//   * `trend_aligned` and the `combined_*` family are reachable in
//     `apply_filter_mask` but have NO entry in dc/obfuscation/map.json, so they
//     have no code and cannot cross this boundary. They are not implemented;
//     an unknown code THROWS rather than returning a mask.
//   * the reference's `strict_filters=False` path returns an all-False mask for
//     an unknown name. This throws instead. A live core that silently gated a
//     typo'd regime to "never fires" is a strategy that is off with no error.
//
// NO NaN / inf SANITISATION. NaN compares False in every predicate here,
// exactly as pandas does, and that is precisely the mechanism by which the 53
// regimes above stay inert. See src/feature_engine.hpp's banner.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "feature_engine.hpp"

namespace agamotto {

// +1 long, -1 short. An enum rather than the reference's "long"/"short"
// strings: a string on this side of the seam is a name, and the two positions
// select genuinely different predicates for the same atom.
enum class Position : int8_t { LONG = 1, SHORT = -1 };

// The widest conjunction BASE_REGIMES can generate is 3 atoms (a 2-atom
// incumbent plus one vol-quantile gate); the deployed stack's widest is 3.
// 8 is a cap on the ABI struct, not a target — a stack wider than this is
// rejected loudly rather than truncated, because a truncated conjunction is a
// LOOSER gate that fires more often and nothing would say so.
constexpr size_t MAX_REGIME_ATOMS = 8;

// Does this core carry a predicate for `code`?
//
// Exists so a stack can be REJECTED AT CONFIGURATION TIME rather than at the
// first warm bar. Agamotto's warmup is 700 15m bars — 7.3 days — so a run that
// discovered an unusable atom on its first panel would have booted clean, sat
// warming for a week and only then said its stack was no good.
//
// It must agree with `atomMask` exactly in both directions, and that is
// ASSERTED rather than maintained by care: tests/regime_parity_driver.cpp
// sweeps every code in [0, 4096) and requires atomIsKnown(c) to be true exactly
// when atomMask(.., c, ..) does not throw. A code accepted here and rejected
// there is a stack that passes validation and then throws on every bar.
bool atomIsKnown(uint16_t code);

// research_filters.allowed_positions, over a conjunction's atoms.
//
// LONG_ONLY / SHORT_ONLY membership is a property of the ATOM; a conjunction
// that mixes one of each is allowed on NEITHER side (the reference returns []).
// Called before any predicate is evaluated, exactly as the reference does, so
// a disallowed position yields an all-False mask rather than an evaluated one.
bool positionAllowed(const std::vector<uint16_t>& atom_codes, Position pos);

// Per-row mask for ONE atom. `panel.rows` entries, 1 = the predicate held.
//
// THROWS std::invalid_argument on an unknown atom code, and std::out_of_range
// (from Table::get) on a column the panel does not carry. Both are the
// reference's `_require_col` discipline: an all-True fallback here is a
// `baseline` regime under another name, which CLAUDE.md removed forever.
std::vector<char> atomMask(const Table& panel, uint16_t code, Position pos);

// Per-row mask for a whole regime: the AND of its atoms, all-False when the
// position is not allowed. This is `apply_filter_mask` for an `_and_`-joined
// name, with the split already done.
//
// An EMPTY atom list throws. The reference would return an all-True mask there
// (`mask if mask is not None else Series(True)`), which is the unconditional
// always-fire regime `baseline` — banned outright.
std::vector<char> regimeMask(const Table& panel,
                             const std::vector<uint16_t>& atom_codes,
                             Position pos);

} // namespace agamotto
