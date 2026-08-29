// PHASE 4 — the model runner. See model_runner.hpp for the artifact contract,
// the mixed-provenance measurement and the two declared divergences.
#include "model_runner.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace agamotto {
namespace {

// A synthetic panel whose ONLY job is to make `engineerFeatures` emit its
// column list. The VALUES are irrelevant — but they must be non-degenerate
// enough that the TA-Lib block does not throw (it refuses an entirely-NaN
// input), so this is the same shape tests/regime_parity_driver.cpp's selftest
// builds and for the same reason.
RawBars probeBars()
{
    RawBars rb;
    rb.open.reserve(PANEL_BARS);
    rb.high.reserve(PANEL_BARS);
    rb.low.reserve(PANEL_BARS);
    rb.close.reserve(PANEL_BARS);
    rb.volume.reserve(PANEL_BARS);
    rb.quote_volume.reserve(PANEL_BARS);
    rb.taker_buy_quote_volume.reserve(PANEL_BARS);
    rb.number_of_trades.reserve(PANEL_BARS);
    for (size_t i = 0; i < PANEL_BARS; ++i) {
        const double px = 100.0 + static_cast<double>(i % 17);
        rb.open.push_back(px);
        rb.high.push_back(px + 1.0);
        rb.low.push_back(px - 1.0);
        rb.close.push_back(px + 0.5);
        rb.volume.push_back(10.0 + static_cast<double>(i % 5));
        rb.quote_volume.push_back(1000.0 + static_cast<double>(i % 7));
        rb.taker_buy_quote_volume.push_back(400.0 + static_cast<double>(i % 3));
        rb.number_of_trades.push_back(50.0 + static_cast<double>(i % 11));
    }
    return rb;
}

std::string readFile(const std::string& path)
{
    std::ifstream f(path.c_str());
    if (!f) {
        throw std::runtime_error("agamotto::loadLinearModel: cannot open " + path);
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    if (f.bad()) {
        throw std::runtime_error("agamotto::loadLinearModel: read error on " + path);
    }
    return ss.str();
}

// Whitespace-delimited token stream. The exporter writes the whole model.txt as
// tokens precisely so a plain `>>` reader suffices; this is that reader, made
// explicit so every "expected X, got Y" can name the file.
class Tokens {
  public:
    Tokens(const std::string& text, std::string path)
      : mIn(text), mPath(std::move(path)) {}

    bool next(std::string& out) { return static_cast<bool>(mIn >> out); }

    std::string require(const char* what)
    {
        std::string t;
        if (!next(t)) {
            throw std::invalid_argument(mPath + ": ended while expecting " + what);
        }
        return t;
    }

    // `<keyword> <value>`; the keyword must match exactly.
    std::string keyword(const char* name)
    {
        const std::string got = require(name);
        if (got != name) {
            throw std::invalid_argument(
                mPath + ": expected keyword '" + name + "', got '" + got + "'");
        }
        return require(name);
    }

    void requireEnd()
    {
        std::string t;
        if (next(t)) {
            throw std::invalid_argument(
                mPath + ": trailing token '" + t + "' after the coefficient block");
        }
    }

    const std::string& path() const { return mPath; }

  private:
    std::istringstream mIn;
    std::string mPath;
};

double toFinite(const std::string& tok, const std::string& path, const char* what)
{
    size_t used = 0;
    double v = 0.0;
    try {
        v = std::stod(tok, &used);
    } catch (const std::exception&) {
        throw std::invalid_argument(path + ": " + what + " '" + tok + "' is not a number");
    }
    if (used != tok.size()) {
        throw std::invalid_argument(
            path + ": " + what + " '" + tok + "' has trailing characters");
    }
    if (!std::isfinite(v)) {
        // A non-finite weight poisons every prediction the regime ever makes,
        // and does it silently on the exact bars the gate lets through.
        throw std::invalid_argument(
            path + ": " + what + " is non-finite ('" + tok + "')");
    }
    return v;
}

long long toCount(const std::string& tok, const std::string& path, const char* what)
{
    size_t used = 0;
    long long v = 0;
    try {
        v = std::stoll(tok, &used);
    } catch (const std::exception&) {
        throw std::invalid_argument(path + ": " + what + " '" + tok + "' is not an integer");
    }
    if (used != tok.size() || v <= 0) {
        throw std::invalid_argument(
            path + ": " + what + " must be a positive integer, got '" + tok + "'");
    }
    return v;
}

} // namespace

// ---------------------------------------------------------------------------

const std::vector<std::string>& canonicalPanelColumns()
{
    // Function-local static: built on first use, which is core construction.
    // One ~55 ms engine pass, once per process.
    static const std::vector<std::string> kCols = engineerFeatures(probeBars()).names;
    return kCols;
}

std::string regimeDirName(const std::vector<uint16_t>& atom_codes, Position pos)
{
    if (atom_codes.empty()) {
        // The empty conjunction is `baseline`, the unconditional always-fire
        // gate removed forever on 2026-06-18. regime_gate refuses it; so does
        // this, rather than naming a directory `_long`.
        throw std::invalid_argument(
            "agamotto::regimeDirName: an EMPTY conjunction has no directory "
            "(it is the always-fire gate `baseline`, removed 2026-06-18)");
    }
    std::string out;
    for (size_t i = 0; i < atom_codes.size(); ++i) {
        if (i) out += "_and_";
        char buf[16];
        // %03u, matching the exporter's directory names: codes below 100 are
        // zero-padded to three digits, wider codes print in full. Verified
        // 2026-08-20 against all 109 exported directories.
        std::snprintf(buf, sizeof(buf), "r%03u", static_cast<unsigned>(atom_codes[i]));
        out += buf;
    }
    out += (pos == Position::LONG) ? "_long" : "_short";
    return out;
}

size_t LinearModel::unitScaleFeatures() const
{
    size_t n = 0;
    for (const double s : scale) {
        if (s == 1.0) ++n;
    }
    return n;
}

double LinearModel::predictRow(const Table& panel, size_t row, int64_t* nan_filled) const
{
    double y = intercept;
    for (size_t i = 0; i < coef.size(); ++i) {
        const std::vector<double>& col = panel.cols.at(column_index[i]);
        if (row >= col.size()) {
            throw std::out_of_range(
                "agamotto::LinearModel::predictRow: row " + std::to_string(row) +
                " is past a " + std::to_string(col.size()) + "-row panel");
        }
        double x = col[row];
        // trading.py:697-700 — NaN in a SELECTED model column becomes 0.0.
        // inf is NOT touched there and is NOT touched here; it propagates into
        // the prediction, which is then reported non-finite rather than hidden.
        if (std::isnan(x)) {
            x = 0.0;
            if (nan_filled != nullptr) ++*nan_filled;
        }
        y += coef[i] * ((x - center[i]) / scale[i]);
    }
    return y;
}

// ---------------------------------------------------------------------------

LinearModel loadLinearModel(const std::string& regime_dir)
{
    const std::string model_path = regime_dir + "/model.txt";
    const std::string scaler_path = regime_dir + "/scaler.txt";
    const std::string features_path = regime_dir + "/features.txt";

    LinearModel m;

    // ---- model.txt --------------------------------------------------------
    Tokens mt(readFile(model_path), model_path);
    {
        std::string first;
        if (!mt.next(first)) {
            throw std::invalid_argument(model_path + ": file is empty");
        }
        if (first != "model_kind") {
            // THE LIGHTGBM REFUSAL. mjolnir's model.txt is LightGBM's native
            // dump and opens with `tree`; agamotto's opens with `model_kind`.
            // Naming both formats here is the difference between a boot that
            // halts and a boot that reads a booster header as coefficients.
            throw std::invalid_argument(
                model_path + ": first token is '" + first + "', not 'model_kind'. "
                "This loader reads agamotto's LINEAR model.txt only; a file "
                "starting with 'tree' is a LightGBM booster dump (mjolnir's "
                "format under the same filename) and must go to the mjolnir "
                "core. Refusing to parse it as coefficients.");
        }
        const std::string kind = mt.require("model_kind value");
        if (kind != "linear") {
            throw std::invalid_argument(
                model_path + ": model_kind is '" + kind + "', not 'linear'");
        }
        const std::string ver = mt.keyword("format_version");
        if (ver != "1") {
            throw std::invalid_argument(
                model_path + ": format_version '" + ver + "' is not supported "
                "(this core reads version 1)");
        }
        const long long n = toCount(mt.keyword("n_features"), model_path, "n_features");
        m.intercept = toFinite(mt.keyword("intercept"), model_path, "intercept");
        const std::string coef_kw = mt.require("'coef'");
        if (coef_kw != "coef") {
            throw std::invalid_argument(
                model_path + ": expected 'coef', got '" + coef_kw + "'");
        }
        m.coef.reserve(static_cast<size_t>(n));
        for (long long i = 0; i < n; ++i) {
            m.coef.push_back(toFinite(mt.require("a coefficient"), model_path,
                                      "coefficient"));
        }
        mt.requireEnd();
    }
    const size_t n_feat = m.coef.size();

    // ---- scaler.txt -------------------------------------------------------
    {
        Tokens st(readFile(scaler_path), scaler_path);
        const long long n = toCount(st.require("the row count"), scaler_path,
                                    "the row count");
        if (static_cast<size_t>(n) != n_feat) {
            throw std::invalid_argument(
                scaler_path + ": declares " + std::to_string(n) + " rows but " +
                model_path + " has " + std::to_string(n_feat) + " coefficients");
        }
        m.center.reserve(n_feat);
        m.scale.reserve(n_feat);
        for (size_t i = 0; i < n_feat; ++i) {
            m.center.push_back(toFinite(st.require("a scaler centre"), scaler_path,
                                        "centre"));
            const double s = toFinite(st.require("a scaler scale"), scaler_path,
                                      "scale");
            if (s == 0.0) {
                // Dividing by it yields NaN or inf on EVERY row, forever. The
                // exporter refuses to emit one; a file carrying one is corrupt.
                throw std::invalid_argument(
                    scaler_path + ": row " + std::to_string(i) + " has scale 0, "
                    "which would divide every prediction into NaN");
            }
            m.scale.push_back(s);
        }
        st.requireEnd();
    }

    // ---- features.txt -----------------------------------------------------
    {
        Tokens ft(readFile(features_path), features_path);
        std::string code;
        while (ft.next(code)) m.feature_codes.push_back(code);
        if (m.feature_codes.size() != n_feat) {
            throw std::invalid_argument(
                features_path + ": names " + std::to_string(m.feature_codes.size()) +
                " feature(s) but " + model_path + " has " + std::to_string(n_feat) +
                " coefficients");
        }
    }

    // ---- resolve code -> panel column index, ONCE -------------------------
    const std::vector<std::string>& cols = canonicalPanelColumns();
    std::set<std::string> seen;
    m.column_index.reserve(n_feat);
    for (const std::string& c : m.feature_codes) {
        if (!seen.insert(c).second) {
            // Mathematically harmless, but it means the artifact is not what
            // the trainer produced (the exporter rejects duplicates too).
            throw std::invalid_argument(
                features_path + ": feature '" + c + "' appears twice");
        }
        size_t j = 0;
        bool found = false;
        for (; j < cols.size(); ++j) {
            if (cols[j] == c) { found = true; break; }
        }
        if (!found) {
            throw std::invalid_argument(
                features_path + ": feature '" + c + "' is not a column of the "
                "engineered panel (" + std::to_string(cols.size()) + " columns). "
                "Resolved at BOOT deliberately: agamotto warms for 700 bars "
                "(7.3 days), so discovering this on the first panel would mean a "
                "week of a run that was never going to score.");
        }
        m.column_index.push_back(j);
    }
    return m;
}

// ---------------------------------------------------------------------------

void ModelBook::load(const std::string& weights_dir,
                     const std::vector<std::string>& regime_dirs)
{
    if (weights_dir.empty()) {
        // A required config key read as "" is a MISSING key, not a request for
        // an unscored run. Failing here is what stops a core that gates,
        // predicts nothing, and looks exactly like a market that never fired.
        throw std::invalid_argument(
            "agamotto::ModelBook::load: weights_dir is empty. It is a REQUIRED "
            "config key; there is no default, because a core with no models "
            "produces a silent no-signal run indistinguishable from a working one");
    }
    if (regime_dirs.empty()) {
        throw std::invalid_argument(
            "agamotto::ModelBook::load: no regimes named — nothing to load");
    }

    std::map<std::string, LinearModel> loaded;
    for (const std::string& r : regime_dirs) {
        if (loaded.count(r) != 0) continue;   // the same regime twice in a stack
        try {
            loaded.emplace(r, loadLinearModel(weights_dir + "/" + r));
        } catch (const std::exception& e) {
            // Named, and fatal. The Python bot's equivalent raise
            // (FileNotFoundError: Regime folder <name> not found) is what caught
            // a real stack/weights mismatch; skipping the row instead would
            // install a leg that silently never trades.
            throw std::invalid_argument(
                std::string("agamotto: regime '") + r + "' has no usable weights "
                "under " + weights_dir + ": " + e.what());
        }
    }
    mModels = std::move(loaded);
    mWeightsDir = weights_dir;
}

const LinearModel& ModelBook::at(const std::string& regime_dir) const
{
    const std::map<std::string, LinearModel>::const_iterator it = mModels.find(regime_dir);
    if (it == mModels.end()) {
        throw std::out_of_range(
            "agamotto::ModelBook: no model for regime '" + regime_dir + "'");
    }
    return it->second;
}

size_t ModelBook::featureCountVariants() const
{
    std::set<size_t> n;
    for (std::map<std::string, LinearModel>::const_iterator it = mModels.begin();
         it != mModels.end(); ++it) {
        n.insert(it->second.featureCount());
    }
    return n.size();
}

size_t ModelBook::minFeatureCount() const
{
    size_t lo = 0;
    bool first = true;
    for (std::map<std::string, LinearModel>::const_iterator it = mModels.begin();
         it != mModels.end(); ++it) {
        const size_t n = it->second.featureCount();
        if (first || n < lo) { lo = n; first = false; }
    }
    return lo;
}

size_t ModelBook::maxFeatureCount() const
{
    size_t hi = 0;
    for (std::map<std::string, LinearModel>::const_iterator it = mModels.begin();
         it != mModels.end(); ++it) {
        const size_t n = it->second.featureCount();
        if (n > hi) hi = n;
    }
    return hi;
}

size_t ModelBook::unitScaleFeatures() const
{
    size_t n = 0;
    for (std::map<std::string, LinearModel>::const_iterator it = mModels.begin();
         it != mModels.end(); ++it) {
        n += it->second.unitScaleFeatures();
    }
    return n;
}

std::string ModelBook::inventory() const
{
    // THE MIXED-PROVENANCE REPORT. `window_2026_07_31` is one directory label
    // over two training runs; nothing else in a live run would say so.
    //
    // MULTI-LINE, '\n'-separated, and each line is kept short ON PURPOSE: the
    // SDK's LOG_INFO formats through a 256-byte stack buffer and TRUNCATES
    // silently past it, so a single long string would lose exactly the tail
    // that names the affected regimes. The caller splits on '\n' and logs each
    // line.
    std::map<size_t, size_t> by_n;
    for (std::map<std::string, LinearModel>::const_iterator it = mModels.begin();
         it != mModels.end(); ++it) {
        ++by_n[it->second.featureCount()];
    }
    std::ostringstream ss;
    ss << mModels.size() << " model(s) from " << mWeightsDir << "; features/regime:";
    for (std::map<size_t, size_t>::const_iterator it = by_n.begin();
         it != by_n.end(); ++it) {
        ss << " " << it->first << "x" << it->second;
    }
    if (by_n.size() > 1) {
        ss << "\nWARNING mixed provenance: " << by_n.size() << " distinct feature "
              "counts in ONE weights directory -- these regimes were NOT all "
              "fitted by the same run, so TOPN_ICS describes only some of them";
    }
    const size_t unit = unitScaleFeatures();
    if (unit > 0) {
        ss << "\nWARNING " << unit << " scaler row(s) have scale == 1.0 EXACTLY: "
              "sklearn's substitute for a ZERO IQR, i.e. a feature that was "
              "CONSTANT over the train window. It enters the model UNSCALED "
              "while its neighbours are divided by their IQR:";
        for (std::map<std::string, LinearModel>::const_iterator it = mModels.begin();
             it != mModels.end(); ++it) {
            for (size_t i = 0; i < it->second.scale.size(); ++i) {
                if (it->second.scale[i] == 1.0) {
                    ss << "\n  unit-scale: " << it->first << " / "
                       << it->second.feature_codes[i] << " (centre "
                       << it->second.center[i] << ")";
                }
            }
        }
    }
    return ss.str();
}

void ModelBook::assertPanelLayout(const Table& panel)
{
    const std::vector<std::string>& cols = canonicalPanelColumns();
    if (panel.names.size() != cols.size()) {
        throw std::invalid_argument(
            "agamotto: the panel has " + std::to_string(panel.names.size()) +
            " columns, the models were resolved against " +
            std::to_string(cols.size()));
    }
    for (size_t j = 0; j < cols.size(); ++j) {
        if (panel.names[j] != cols[j]) {
            throw std::invalid_argument(
                "agamotto: panel column " + std::to_string(j) + " is '" +
                panel.names[j] + "', the models were resolved against '" +
                cols[j] + "' — every model would read a neighbouring column");
        }
    }
}

} // namespace agamotto
