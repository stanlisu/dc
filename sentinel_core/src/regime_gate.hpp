#pragma once
// Regime gate. PRIVATE — predicates are IP.
//
// This is the piece the port plan flags as most error-prone, for one specific
// reason: several predicates are QUANTILE-based and the quantile is taken over
// the WHOLE PANEL. Evaluate them on a one-row frame and the quantile equals the
// value, so `>` is always false and the regime silently never fires — a
// disabled strategy that looks like a quiet market. Always pass the full panel.
#include <string>
#include <vector>

namespace mjolnir {

class FeaturePanel;   // thin view over the feature engine's output

// Column-name -> column lookup over the feature engine result.
class FeaturePanel {
  public:
    FeaturePanel(const std::vector<std::string>& names,
                 const std::vector<std::vector<double>>& cols);
    bool has(const std::string& name) const;
    const std::vector<double>& get(const std::string& name) const;
    size_t rows() const { return mRows; }

  private:
    const std::vector<std::string>& mNames;
    const std::vector<std::vector<double>>& mCols;
    size_t mRows{0};
};

// Evaluate a regime expression into a per-row boolean mask.
//
// `name` accepts EITHER a code (rNNN) or a real name, mirroring the reference's
// tolerant decode — the live stack still carries real names while coded stacks
// are rolling out, and rejecting either form would break one of them.
// Supports `_and_` / `_or_` composition and a trailing _long/_short which is
// stripped (position is passed separately).
//
// Throws on an unknown regime name: a silently-true mask would turn a typo into
// an unconditional always-fire regime.
std::vector<char> applyFilterMask(const FeaturePanel& panel,
                                  const std::string& name,
                                  const std::string& position);

// pandas Series.quantile(q) — linear interpolation, NaN excluded.
double quantileLinear(const std::vector<double>& x, double q);

// Decode a coded regime (rNNN) to its real name; returns the input unchanged if
// it is not a code (tolerant decode).
std::string decodeRegimeTolerant(const std::string& name);

} // namespace mjolnir
