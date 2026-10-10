#pragma once
// ABI 6 (orb): which CONTEXT-timeframe row a base bar reads. PRIVATE.
//
// orb research.py `_align_timeframes` joins each context timeframe onto the base
// rows with pd.merge_asof(left = base OPEN time, right = context close_timestamp
// = open + tf, direction = "backward"): the newest context bar whose close is
// at or before the base bar's OPEN. At the decision taken when base bar
// [T, T + base) closes, the context bar closing at T + base is therefore NOT
// read yet -- a one-base-bar lag that research and knull orb share, reproduced
// here rather than "fixed".
//
// The row must also be the bar closing at floor(T / tf) * tf. An older one is
// STALE (the newer bar has not arrived) and the caller treats it as "the regime
// does not fire": live would otherwise trade a regime on a context bar research
// never paired with this base bar.
//
// One function, used by RealCore and by tests/orb_parity_driver.cpp, so the
// parity run grades the code that ships.
#include <algorithm>
#include <cstdint>
#include <vector>

namespace agamotto {

enum class AsofStatus { FRESH, STALE, MISSING };

// `close_ms` ascending. Returns the row index when FRESH, -1 otherwise.
// `found_close_ms` (optional) receives the close of the row the as-of found,
// 0 when none.
inline int contextAsofRow(const std::vector<int64_t>& close_ms, int64_t base_open_ms,
                          int64_t tf_ms, AsofStatus* status, int64_t* found_close_ms)
{
    if (found_close_ms) *found_close_ms = 0;
    const auto it = std::upper_bound(close_ms.begin(), close_ms.end(), base_open_ms);
    if (it == close_ms.begin()) {
        if (status) *status = AsofStatus::MISSING;
        return -1;
    }
    const auto row = it - 1;
    if (found_close_ms) *found_close_ms = *row;
    const int64_t need = (base_open_ms / tf_ms) * tf_ms;
    if (*row != need) {
        if (status) *status = AsofStatus::STALE;
        return -1;
    }
    if (status) *status = AsofStatus::FRESH;
    return static_cast<int>(row - close_ms.begin());
}

} // namespace agamotto
