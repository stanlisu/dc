#pragma once
// Feature computation. PRIVATE — this is the alpha.
//
// PANEL-BASED BY DESIGN (for now): computes over the whole bar window exactly
// like the reference, because M1's gate is numerical parity and the panel form
// is what parity is measured against. The incremental O(1) ring-buffer form is
// the actual latency win and comes AFTER parity is green — optimising before
// the reference is matched would leave no trustworthy baseline to optimise
// against.
//
// pandas semantics that must be reproduced exactly (each one silently changes
// values if approximated):
//   * diff()/shift(1)/pct_change() -> NaN in the first slot(s)
//   * rolling(w, min_periods=1).mean() -> expanding until the window fills
//   * rolling(w, min_periods=1).std()  -> SAMPLE std (ddof=1) => NaN at n==1
//   * the point-in-time +1 shift applied to book/snapshot/derivative columns
//     BEFORE rolling stats and targets
//   * final sanitisation: +/-inf -> 0.0, NaN -> 0.0 on feature columns only
#include "bar_builder.hpp"

#include <string>
#include <utility>
#include <vector>

namespace mjolnir {

class FeatureEngine {
  public:
    FeatureEngine(std::vector<int> windows, int bar_sec, int target_sec);

    // Column-major result: names[j] holds cols[j], each of size bars.size().
    void compute(const std::vector<Bar>& bars,
                 std::vector<std::string>& names,
                 std::vector<std::vector<double>>& cols) const;

    // Whether cycle features apply (target coarser than the base bar).
    bool boundaryMode() const { return mTargetSec > mBarSec; }

  private:
    std::vector<int> mWindows;
    int mBarSec;
    int mTargetSec;
};

// --- pandas-equivalent primitives (exposed for unit testing) --------------
namespace pdops {

// x.diff(): out[0] = NaN, out[i] = x[i] - x[i-1]
std::vector<double> diff(const std::vector<double>& x);
// x.shift(n): first n become NaN
std::vector<double> shift(const std::vector<double>& x, int n);
// x.pct_change(fill_method=None): out[0] = NaN, out[i] = x[i]/x[i-1] - 1
std::vector<double> pctChange(const std::vector<double>& x, int periods = 1);
// x.rolling(w, min_periods=mp).mean() — NaN while the non-NaN count < mp.
// mp is NOT always 1: the key-signal rolling stats use max(1, w//4), and
// defaulting it to 1 silently fills the first w/4 rows with values the
// reference leaves NaN (then zeroes).
std::vector<double> rollMean(const std::vector<double>& x, int w, int mp = 1);
// x.rolling(w, min_periods=mp).std() — SAMPLE std (ddof=1); NaN when the
// count is below mp or below 2.
std::vector<double> rollStd(const std::vector<double>& x, int w, int mp = 1);

} // namespace pdops

namespace talib_block {
// Appends the TA-Lib indicator columns. Implemented in talib_block.cpp against
// libta-lib directly, so values match the reference's wrapper exactly.
void compute(const std::vector<double>& close, const std::vector<double>& high,
             const std::vector<double>& low, const std::vector<double>& volume,
             std::vector<std::pair<std::string, std::vector<double>>>& out);
} // namespace talib_block

} // namespace mjolnir
