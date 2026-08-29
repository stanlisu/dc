#pragma once
// PHASE 4 — the model runner. PRIVATE.
//
// Loads the three text files marvel's `gauntlet/export_agamotto_sentinel_weights.py`
// writes per regime and evaluates them on a row of the engineered panel.
//
// ---------------------------------------------------------------------------
// THE ARTIFACT CONTRACT, RESTATED HERE BECAUSE THIS FILE IS ITS ONLY READER
// ---------------------------------------------------------------------------
//
//   model.txt      model_kind linear
//                  format_version 1
//                  n_features N
//                  intercept <double>
//                  coef
//                  <w0> ... <w_{N-1}>          (each %.17g, one per line)
//   scaler.txt     <N>
//                  <center_i> <scale_i>        (N lines; RobustScaler)
//   features.txt   N CODED feature names, in the model's input order
//
//   y = intercept + SUM_i coef[i] * ((x[i] - center[i]) / scale[i])
//
// That is the WHOLE model. No link function, no per-tree walk, and NO CLIPPING
// (see the declared divergence below). `REVERSE` and the threshold gate are
// applied DOWNSTREAM by the strategy out of algo_params; they are not the
// weights' business and are not applied here.
//
// *** `model_kind` IS THE FIRST TOKEN SO A LIGHTGBM DUMP CANNOT BE LOADED. ***
// sentinel_core (mjolnir) reads a `model.txt` that is LightGBM's own native
// text dump and opens with the token `tree`. Same filename, completely
// different format. `loadLinearModel` REFUSES any file whose first token is not
// `model_kind`, so pointing agamotto at a mjolnir weight directory fails at
// boot with a message naming both formats instead of parsing a booster's header
// as coefficients and predicting numbers that look plausible.
//
// ---------------------------------------------------------------------------
// N IS PER REGIME AND IS **NOT** 5, EVEN THOUGH `TOPN_ICS` IS.
// ---------------------------------------------------------------------------
// `window_2026_07_31` is NOT one training run. Measured 2026-08-20 over all 109
// exported regime directories:
//
//   53 regimes carry  5-feature models, fitted 2026-08-16, window_id 33
//   56 regimes carry 16-feature models, fitted 2026-08-13, window_id 33 AND 34
//
// and across the 62 rows of the DEPLOYED stack the split falls exactly on the
// fault line that matters:
//
//   *** ALL 53 vol-quantile-gated (inert) regimes are 5-feature. ***
//   *** ALL  9 FIRABLE regimes are 16-feature. ***
//
// So a runner that hardcoded `TOPN_ICS = 5` would not fail quietly on a corner
// case — it would be wrong on every regime that can actually trade, and right
// on every regime that cannot. `n_features` is therefore read from the artifact
// per regime and `features.txt` is the sole authority on which columns to take.
// `ModelBook::inventory()` prints the mixture at BOOT so the operator sees it
// as a fact about the artifacts rather than discovering it from a bad fill.
//
// ---------------------------------------------------------------------------
// CODES ONLY, and index resolution happens ONCE.
// ---------------------------------------------------------------------------
// `features.txt` carries obfuscation codes (`f065`); so does the panel. Each
// code is resolved to a COLUMN INDEX at load time against
// `canonicalPanelColumns()`, and a code the panel does not carry is a BOOT
// error, never a per-bar one: agamotto's warmup is 700 15m bars (7.3 days), so
// a runner that discovered a missing column on its first panel would have
// booted clean, warmed for a week and only then said its weights were unusable.
//
// No real feature name and no real regime name appears in this file, and the
// regime DIRECTORY name is reconstructed from the atom codes the ABI carries
// (`{29,1,73}`,LONG -> `r029_and_r001_and_r073_long`) rather than being passed
// in as a string, so `strings libagamotto_core.so` still recovers nothing.
//
// ---------------------------------------------------------------------------
// DECLARED DIVERGENCE #1 — trading.py's PREDICTION CLIP IS NOT REPRODUCED.
// ---------------------------------------------------------------------------
// dc/agamotto_pkg/src/agamotto/trading.py:705-713 does, after predicting:
//
//     if np.abs(preds).max() > 1.0:
//         logger.warning("prediction overflow ... clipping to [-1, 1]")
//         preds = np.clip(preds, -1.0, 1.0)
//
// That is a `np.clip` on a DERIVED quantity, which CLAUDE.md bans outright: an
// out-of-range prediction is the degenerate-scaler bug signalling itself, and
// clamping it to 1.0 converts the signal into a plausible in-range number. It
// is ALSO not part of the exported contract — `utils.weights_io.RegimeArtifacts
// .predict`, the loader this port's parity gate grades against, does not clip —
// so reproducing it here would make the C++ disagree with the reference the
// gate uses. REPORTED, NOT FIXED (the fix belongs in trading.py). The practical
// effect: a |y_pred| above 1.0 rides out of this core intact, and the strategy
// sees the real number. Since the deployed thresholds are ~5.8e-4, clipping
// never changed WHETHER a leg fired; it changed only the reported magnitude and
// therefore any ranking built on it.
//
// ---------------------------------------------------------------------------
// DECLARED DIVERGENCE #2 — SIX SCALER ROWS HAVE scale == 1.0 FROM A ZERO IQR.
// ---------------------------------------------------------------------------
// sklearn's `_handle_zeros_in_scale` substitutes 1.0 for a RobustScaler
// `scale_` entry whose train-window IQR is exactly zero, i.e. a feature that
// was CONSTANT over the training period. Measured across the 109 exported
// regimes, six rows are affected and all six are the same column, `f089`:
//
//     r029_and_r019_and_r074_long   f089  center=100  scale=1   (deployed, inert)
//     r069_and_r001_long            f089  center=100  scale=1   (DEPLOYED, FIRABLE)
//     r069_and_r019_long            f089  center=100  scale=1
//     r069_and_r045_long            f089  center=100  scale=1
//     r069_and_r045_short           f089  center=0    scale=1
//     r069_and_r066_long            f089  center=100  scale=1
//
// `f089` is bounded in [0, 100], so a constant train window pins it at a rail.
// The consequence is NOT a divide-by-zero — it is that the feature enters the
// model UNSCALED while every other feature is divided by its own IQR, so
// `coef * (x - 100)` swings over the full [-100, 0] range at whatever weight
// the fit assigned. This is exported faithfully (the numbers ARE the deployed
// model) but it is COUNTED and named at boot — `unit_scale_features` /
// `inventory()` — because a scale of exactly 1.0 next to IQR-scaled neighbours
// is a fitted-on-a-constant artifact, not a modelling choice, and
// `r069_and_r001_long` is the regime the live hydra bot is running.
//
// ---------------------------------------------------------------------------
// NaN HANDLING IS trading.py's, AND ONLY trading.py's.
// ---------------------------------------------------------------------------
// trading.py:697-700 fills NaN with 0.0 across the SELECTED MODEL COLUMNS of the
// single scored row, AFTER the regime gate has run, and does not touch inf.
// Reproduced exactly: a NaN feature contributes `coef * ((0 - center) / scale)`,
// which is emphatically not "no contribution", and a +/-inf feature propagates
// into the prediction. Both are COUNTED (`nan_features_filled`,
// `nonfinite_predictions`) rather than smoothed, because a bar predicted off a
// filled column is a different kind of number from one predicted off live data.
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "feature_engine.hpp"
#include "regime_gate.hpp"

namespace agamotto {

// The panel's column codes in the order `engineerFeatures` emits them.
//
// Obtained by RUNNING the engine once, on a synthetic PANEL_BARS-wide panel,
// rather than from a list maintained alongside it. A hand-kept list is one
// edit away from disagreeing with the engine, and the disagreement would show
// up as a model reading the WRONG COLUMN — a plausible number for the wrong
// feature, which nothing downstream can detect. Computed once (function-local
// static) and costs one ~55 ms engine pass at core construction.
const std::vector<std::string>& canonicalPanelColumns();

// `{29, 1, 73}` + LONG -> `r029_and_r001_and_r073_long`.
//
// The core is handed CODES (RegimeSpec), while the exporter names directories
// by the coded regime, so the two are bridged HERE by construction rather than
// by carrying a directory string across the ABI. Verified 2026-08-20 to
// reproduce all 109 exported directory names and all 62 deployed stack rows
// exactly. Atom ORDER is preserved, never sorted: `r029_and_r001_...` and
// `r001_and_r029_...` are different directories.
std::string regimeDirName(const std::vector<uint16_t>& atom_codes, Position pos);

// One regime's linear model, with its feature codes already resolved to panel
// column indices.
struct LinearModel {
    std::vector<std::string> feature_codes;  // coded, model input order
    std::vector<size_t> column_index;        // into canonicalPanelColumns()
    std::vector<double> center;
    std::vector<double> scale;
    std::vector<double> coef;
    double intercept{0.0};

    size_t featureCount() const { return feature_codes.size(); }

    // Scaler entries whose scale is EXACTLY 1.0 — sklearn's substitute for a
    // zero IQR (a constant train feature). See divergence #2 above.
    size_t unitScaleFeatures() const;

    // y for ONE row of `panel`.
    //
    // `nan_filled`, when non-null, is INCREMENTED once per feature cell that
    // was NaN and got trading.py's 0.0. Never reset here — the caller owns the
    // run total.
    //
    // Throws std::out_of_range if `row` is past the panel's end. It does NOT
    // re-resolve columns: `column_index` was fixed at load, and the caller is
    // responsible for having proved the panel's layout is the canonical one
    // (ModelBook::assertPanelLayout).
    double predictRow(const Table& panel, size_t row, int64_t* nan_filled) const;
};

// Reads model.txt / scaler.txt / features.txt out of ONE regime directory.
//
// Throws std::invalid_argument on anything malformed, including:
//   * a first token that is not `model_kind` (a LightGBM dump, see above)
//   * model_kind != linear, or format_version != 1
//   * n_features disagreeing between the three files
//   * a non-finite coefficient / centre / scale, or a scale of exactly 0
//   * a duplicate feature code
//   * a feature code that is not a column of `canonicalPanelColumns()`
// Throws std::runtime_error when a file is missing or unreadable.
//
// There is deliberately no partial load and no default for anything: a model
// that predicted from a substituted coefficient would produce a number, and a
// number is indistinguishable from a working one.
LinearModel loadLinearModel(const std::string& regime_dir);

// Every regime's model, keyed by coded directory name.
class ModelBook {
  public:
    // Loads EXACTLY the regimes named, out of `weights_dir`. Nothing is
    // scanned or discovered: a directory present on disk but absent from the
    // stack is not loaded (it is not traded), and a stack entry with no
    // directory THROWS naming the path — which is the same failure the Python
    // bot raises (`FileNotFoundError: Regime folder r060_and_r075_long not
    // found`) and which caught a real stack/weights mismatch.
    //
    // Throws std::invalid_argument on an empty weights_dir (a missing required
    // config key must not read as "no models wanted") and propagates every
    // loadLinearModel failure with the regime named.
    void load(const std::string& weights_dir, const std::vector<std::string>& regime_dirs);

    bool empty() const { return mModels.empty(); }
    size_t size() const { return mModels.size(); }

    // Throws std::out_of_range when absent — never a default-constructed model,
    // whose all-zero coefficients would predict a confident 0.0 forever.
    const LinearModel& at(const std::string& regime_dir) const;

    // ---- the boot-time provenance report --------------------------------
    // One line per distinct feature-count, plus the unit-scale tally. Printed
    // by the strategy at boot: `window_2026_07_31` mixes two training runs, and
    // that is a property of the ARTIFACTS which nothing else in the run would
    // reveal. Codes only.
    std::string inventory() const;

    size_t featureCountVariants() const;   // >1 means MIXED PROVENANCE
    size_t minFeatureCount() const;
    size_t maxFeatureCount() const;
    size_t unitScaleFeatures() const;      // summed over every loaded model

    // The panel a prediction is about to be taken from must have the layout the
    // column indices were resolved against. Checked once per panel, not once
    // per model per row: a panel whose columns moved would otherwise feed every
    // model its neighbour's numbers.
    //
    // Throws std::invalid_argument on any mismatch.
    static void assertPanelLayout(const Table& panel);

  private:
    std::map<std::string, LinearModel> mModels;
    std::string mWeightsDir;
};

} // namespace agamotto
