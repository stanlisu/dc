// PHASE 4 gate driver. Reads one symbol's raw OHLCV panel as CSV on stdin,
// engineers the 65-column panel, loads the named regimes' weights out of a
// weights directory, and prints BOTH the panel and the per-regime y_pred for
// EVERY row.
//
// WHY BOTH, again. tests/model_parity.py must drive the DEPLOYED sklearn loader
// (utils.weights_io.load_regime(...).predict — the path the bot and tesseract
// use) over the *SAME* engineered panel this binary scored, not over a panel
// Python engineered alongside. Feature parity is already gated separately at
// 1e-9 relative; re-engineering here would fold that tolerance into a
// comparison that is supposed to be about the dot product alone.
//
// WHY EVERY ROW. The core scores only the panel's last row, so a live-shaped
// harness would compare FIVE numbers per regime — nowhere near enough to
// separate a correct prediction from one that is right near the mean and wrong
// in the tails, or that gets a coefficient's SIGN wrong on a rarely-large
// feature. `predictRow` is row-independent, so grading all 699 rows costs one
// extra pass and multiplies the evidence by 699. The last row is still what
// live uses and is reported separately.
//
//   --weights <dir>    the export_agamotto_sentinel_weights.py output dir
//   --regimes <spec>   comma-separated, each `code[.code]*:L|S`
//                      e.g. 60.75:L,29.66.73:L,69.65:S
//                      CODES ONLY — the same thing that crosses ICore. The
//                      driver reconstructs the coded DIRECTORY name from them
//                      through the same regimeDirName() the core uses, so a
//                      naming bug fails here rather than only in production.
//   --no-scaler        NEGATIVE CONTROL: predict WITHOUT applying the scaler,
//                      i.e. y = intercept + sum(coef * x_raw). The gate must go
//                      RED. Without it, a harness cannot tell an engine that
//                      applies (x - center)/scale from one that ignores the
//                      scaler entirely on features whose centre is near 0 and
//                      whose scale is near 1.
//   --perturb-coef <regime_index>:<feature_index>:<delta>
//                      NEGATIVE CONTROL: add `delta` to one coefficient of one
//                      loaded model. The gate must go RED for that regime and
//                      stay green for the others.
//   --selftest         the refuse-to-load tests (below); no stdin, no panel.
//
// Output:
//   #panel
//   <65 column codes>
//   <PANEL_BARS rows, %.17g>
//   #preds
//   <the --regimes specs, echoed verbatim as the header>
//   <PANEL_BARS rows of %.17g>
//   #meta
//   <one line per regime: spec dir n_features nan_filled>
//
// NaN and inf ride out untouched, exactly as the other drivers emit them.
#include "../src/feature_engine.hpp"
#include "../src/model_runner.hpp"
#include "../src/regime_gate.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace agamotto;

namespace {

std::vector<std::string> split(const std::string& s, char d)
{
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, d)) out.push_back(item);
    return out;
}

double toDouble(const std::string& s)
{
    const char* p = s.c_str();
    char* end = nullptr;
    const double v = std::strtod(p, &end);
    if (end == p)
        throw std::invalid_argument("model_parity_driver: unparseable value '" + s + "'");
    return v;
}

struct ParsedSpec {
    std::vector<uint16_t> atoms;
    Position pos{Position::LONG};
    std::string raw;
};

// Identical parse to tests/regime_parity_driver.cpp — deliberately duplicated
// rather than shared, because a helper header between two parity drivers is a
// place for one bug to make both of them agree.
ParsedSpec parseSpec(const std::string& s)
{
    ParsedSpec out;
    out.raw = s;
    const size_t colon = s.find(':');
    if (colon == std::string::npos)
        throw std::invalid_argument("regime spec '" + s + "' has no :L/:S position");
    const std::string pos = s.substr(colon + 1);
    if (pos == "L") out.pos = Position::LONG;
    else if (pos == "S") out.pos = Position::SHORT;
    else throw std::invalid_argument("regime spec '" + s + "': position must be L or S");
    for (const std::string& tok : split(s.substr(0, colon), '.')) {
        if (tok.empty())
            throw std::invalid_argument("regime spec '" + s + "' has an empty atom");
        char* end = nullptr;
        const long v = std::strtol(tok.c_str(), &end, 10);
        if (end == tok.c_str() || *end != '\0' || v <= 0 || v > 65535)
            throw std::invalid_argument("regime spec '" + s + "': bad atom '" + tok + "'");
        out.atoms.push_back(static_cast<uint16_t>(v));
    }
    if (out.atoms.empty())
        throw std::invalid_argument("regime spec '" + s + "' has no atoms");
    return out;
}

// y WITHOUT the scaler — the `--no-scaler` control's kernel. Deliberately here
// in the driver rather than as a flag inside LinearModel: the shipping code
// must have no "skip the scaler" path at all, or the control would be testing a
// branch that exists only because the control needs it.
double predictRowUnscaled(const LinearModel& m, const Table& panel, size_t row)
{
    double y = m.intercept;
    for (size_t i = 0; i < m.coef.size(); ++i) {
        double x = panel.cols.at(m.column_index[i])[row];
        if (x != x) x = 0.0;      // the same NaN fill; only the scaler is dropped
        y += m.coef[i] * x;
    }
    return y;
}

// ---------------------------------------------------------------------------
// THE REFUSE-TO-LOAD TESTS.
//
// Every one of these is a way a wrong weight tree could be loaded and PREDICT
// ANYWAY — producing numbers, which is the failure mode with no symptom. They
// build the malformed artifact on disk and require loadLinearModel to throw.
// ---------------------------------------------------------------------------
struct SelfTest {
    int failures{0};

    void expectThrow(const char* what, const std::string& dir)
    {
        try {
            (void)loadLinearModel(dir);
            std::printf("  FAIL: %s was LOADED instead of refused\n", what);
            ++failures;
        } catch (const std::exception& e) {
            std::printf("  ok:   %s refused -- %s\n", what, e.what());
        }
    }
};

void writeFile(const std::string& path, const std::string& text)
{
    std::ofstream f(path.c_str());
    if (!f) throw std::runtime_error("cannot write " + path);
    f << text;
}

// A minimal VALID export, so each malformed case differs from a working one in
// exactly one respect.
void writeGood(const std::string& dir, const std::string& feature_code)
{
    writeFile(dir + "/model.txt",
              "model_kind linear\nformat_version 1\nn_features 2\n"
              "intercept 0.5\ncoef\n1.25\n-2.5\n");
    writeFile(dir + "/scaler.txt", "2\n0.1 2\n0.2 4\n");
    writeFile(dir + "/features.txt", feature_code + "\n" + feature_code + "x\n");
}

int selftest(const std::string& tmp)
{
    SelfTest t;

    // The panel's real column codes, so "present" and "absent" are facts rather
    // than assumptions.
    const std::vector<std::string>& cols = canonicalPanelColumns();
    if (cols.size() < 2) {
        std::printf("  FAIL: canonicalPanelColumns() returned %zu columns\n", cols.size());
        return 1;
    }
    std::printf("  ok:   canonicalPanelColumns() = %zu columns, first '%s'\n",
                cols.size(), cols.front().c_str());

    const std::string d = tmp + "/mp_selftest";
    ::system(("rm -rf " + d + " && mkdir -p " + d).c_str());

    // 1. A LIGHTGBM model.txt. mjolnir's native booster dump under the SAME
    //    filename; its first token is `tree`. Loading it as coefficients would
    //    parse a booster header into weights and predict plausible garbage.
    writeFile(d + "/model.txt",
              "tree\nversion=v3\nnum_class=1\nnum_tree_per_iteration=1\n"
              "max_feature_idx=4\nobjective=regression\n");
    writeFile(d + "/scaler.txt", "2\n0.1 2\n0.2 4\n");
    writeFile(d + "/features.txt", cols[0] + "\n" + cols[1] + "\n");
    t.expectThrow("a LightGBM model.txt (first token 'tree')", d);

    // 2. A MISSING regime directory.
    t.expectThrow("a missing regime directory", d + "/does_not_exist");

    // 3. A features.txt code the PANEL DOES NOT CARRY. This must be a BOOT
    //    error: agamotto warms 700 bars (7.3 days), so a runtime discovery
    //    means a week of a run that was never going to score.
    writeGood(d, cols[0]);
    writeFile(d + "/features.txt", cols[0] + "\nf99999\n");
    t.expectThrow("a features.txt code absent from the panel", d);

    // 4. Coverage of the remaining ways a corrupt artifact predicts anyway.
    writeGood(d, cols[0]);
    writeFile(d + "/features.txt", cols[0] + "\n" + cols[1] + "\n");
    try {
        const LinearModel m = loadLinearModel(d);
        std::printf("  ok:   the CONTROL (a well-formed 2-feature export) LOADS"
                    " -- %zu features, intercept %g\n", m.featureCount(), m.intercept);
    } catch (const std::exception& e) {
        std::printf("  FAIL: the well-formed control was REFUSED -- %s\n", e.what());
        ++t.failures;
    }

    writeFile(d + "/scaler.txt", "2\n0.1 0\n0.2 4\n");
    t.expectThrow("a scaler row with scale == 0 (every prediction would be NaN)", d);

    writeGood(d, cols[0]);
    writeFile(d + "/features.txt", cols[0] + "\n" + cols[1] + "\n");
    writeFile(d + "/scaler.txt", "3\n0.1 2\n0.2 4\n0.3 8\n");
    t.expectThrow("a scaler.txt row count that disagrees with n_features", d);

    writeGood(d, cols[0]);
    writeFile(d + "/features.txt", cols[0] + "\n");
    t.expectThrow("a features.txt shorter than n_features", d);

    writeGood(d, cols[0]);
    writeFile(d + "/features.txt", cols[0] + "\n" + cols[0] + "\n");
    t.expectThrow("a DUPLICATE feature code", d);

    writeGood(d, cols[0]);
    writeFile(d + "/features.txt", cols[0] + "\n" + cols[1] + "\n");
    writeFile(d + "/model.txt",
              "model_kind linear\nformat_version 2\nn_features 2\n"
              "intercept 0.5\ncoef\n1.25\n-2.5\n");
    t.expectThrow("an unsupported format_version", d);

    writeFile(d + "/model.txt",
              "model_kind linear\nformat_version 1\nn_features 2\n"
              "intercept 0.5\ncoef\n1.25\nnan\n");
    t.expectThrow("a non-finite coefficient", d);

    writeFile(d + "/model.txt",
              "model_kind linear\nformat_version 1\nn_features 2\n"
              "intercept 0.5\ncoef\n1.25\n-2.5\n99.0\n");
    t.expectThrow("a trailing token after the coefficient block", d);

    // 5. regimeDirName: the codes -> directory bridge. Wrong here means every
    //    regime looks missing, or worse, one regime loads ANOTHER's weights.
    const std::string got = regimeDirName({29, 1, 73}, Position::LONG);
    if (got != "r029_and_r001_and_r073_long") {
        std::printf("  FAIL: regimeDirName({29,1,73},LONG) = '%s'\n", got.c_str());
        ++t.failures;
    } else {
        std::printf("  ok:   regimeDirName({29,1,73},LONG) = '%s' (order PRESERVED,"
                    " never sorted)\n", got.c_str());
    }
    const std::string got_s = regimeDirName({69}, Position::SHORT);
    if (got_s != "r069_short") {
        std::printf("  FAIL: regimeDirName({69},SHORT) = '%s'\n", got_s.c_str());
        ++t.failures;
    }
    try {
        (void)regimeDirName({}, Position::LONG);
        std::printf("  FAIL: an EMPTY conjunction produced a directory name\n");
        ++t.failures;
    } catch (const std::exception&) {
        std::printf("  ok:   an EMPTY conjunction (the always-fire gate) has no"
                    " directory\n");
    }

    // 6. ModelBook: an empty weights_dir is a MISSING required key, not a
    //    request for an unscored run.
    try {
        ModelBook b;
        b.load("", {"r069_long"});
        std::printf("  FAIL: an EMPTY weights_dir was accepted\n");
        ++t.failures;
    } catch (const std::exception&) {
        std::printf("  ok:   an EMPTY weights_dir is refused\n");
    }

    ::system(("rm -rf " + d).c_str());
    std::printf("\n%s (%d failure%s)\n", t.failures ? "FAILED" : "PASSED",
                t.failures, t.failures == 1 ? "" : "s");
    return t.failures ? 1 : 0;
}

// ---------------------------------------------------------------------------

int run(const std::string& weights, const std::string& regimes_arg,
        bool no_scaler, const std::string& perturb)
{
    std::vector<ParsedSpec> specs;
    for (const std::string& s : split(regimes_arg, ',')) {
        if (!s.empty()) specs.push_back(parseSpec(s));
    }
    if (specs.empty())
        throw std::invalid_argument("model_parity_driver: --regimes named nothing");

    std::vector<std::string> dirs;
    dirs.reserve(specs.size());
    for (const ParsedSpec& s : specs) dirs.push_back(regimeDirName(s.atoms, s.pos));

    ModelBook book;
    book.load(weights, dirs);

    // The perturbation control operates on a COPY of the loaded model, so the
    // book itself is never mutated and the other regimes are provably untouched.
    std::map<size_t, LinearModel> patched;
    if (!perturb.empty()) {
        const std::vector<std::string> f = split(perturb, ':');
        if (f.size() != 3)
            throw std::invalid_argument(
                "--perturb-coef wants <regime_index>:<feature_index>:<delta>");
        const size_t ri = static_cast<size_t>(std::strtoul(f[0].c_str(), nullptr, 10));
        const size_t fi = static_cast<size_t>(std::strtoul(f[1].c_str(), nullptr, 10));
        const double dv = toDouble(f[2]);
        if (ri >= specs.size())
            throw std::invalid_argument("--perturb-coef: regime index out of range");
        LinearModel m = book.at(dirs[ri]);
        if (fi >= m.coef.size())
            throw std::invalid_argument("--perturb-coef: feature index out of range");
        m.coef[fi] += dv;
        patched[ri] = m;
        std::fprintf(stderr, "model_parity_driver: PERTURBED %s coef[%zu] by %g\n",
                     dirs[ri].c_str(), fi, dv);
    }

    std::string header;
    if (!std::getline(std::cin, header))
        throw std::invalid_argument("model_parity_driver: empty input, no header");
    const std::vector<std::string> cols = split(header, ',');

    std::map<std::string, std::vector<double>> raw;
    for (const auto& c : cols) raw[c] = {};

    std::string line;
    size_t rows = 0;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        const std::vector<std::string> f = split(line, ',');
        if (f.size() != cols.size())
            throw std::invalid_argument(
                "model_parity_driver: row " + std::to_string(rows) + " has " +
                std::to_string(f.size()) + " fields, header has " +
                std::to_string(cols.size()));
        for (size_t j = 0; j < cols.size(); ++j) raw[cols[j]].push_back(toDouble(f[j]));
        ++rows;
    }

    RawBars bars;
    bars.open = raw["open"];
    bars.high = raw["high"];
    bars.low = raw["low"];
    bars.close = raw["close"];
    bars.volume = raw["volume"];
    bars.quote_volume = raw["quote_volume"];
    bars.taker_buy_quote_volume = raw["taker_buy_quote_volume"];
    bars.number_of_trades = raw["number_of_trades"];

    const Table t = engineerFeatures(bars);

    // The SAME guard the core runs before it scores anything. Proving it here
    // means the harness cannot pass on a panel the core would have refused.
    ModelBook::assertPanelLayout(t);

    std::printf("#panel\n");
    for (size_t j = 0; j < t.names.size(); ++j)
        std::printf("%s%c", t.names[j].c_str(), j + 1 == t.names.size() ? '\n' : ',');
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < t.cols.size(); ++j)
            std::printf("%.17g%c", t.cols[j][i], j + 1 == t.cols.size() ? '\n' : ',');
    }

    std::vector<int64_t> nan_filled(specs.size(), 0);
    std::vector<std::vector<double>> preds(specs.size());
    for (size_t s = 0; s < specs.size(); ++s) {
        const std::map<size_t, LinearModel>::const_iterator p = patched.find(s);
        const LinearModel& m = (p != patched.end()) ? p->second : book.at(dirs[s]);
        preds[s].reserve(rows);
        for (size_t i = 0; i < rows; ++i) {
            preds[s].push_back(no_scaler ? predictRowUnscaled(m, t, i)
                                         : m.predictRow(t, i, &nan_filled[s]));
        }
    }

    std::printf("#preds\n");
    for (size_t j = 0; j < specs.size(); ++j)
        std::printf("%s%c", specs[j].raw.c_str(), j + 1 == specs.size() ? '\n' : ',');
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < specs.size(); ++j)
            std::printf("%.17g%c", preds[j][i], j + 1 == specs.size() ? '\n' : ',');
    }

    // Per-regime provenance, so the harness grades the MIXTURE rather than
    // taking the feature counts on faith.
    std::printf("#meta\n");
    for (size_t j = 0; j < specs.size(); ++j) {
        const LinearModel& m = book.at(dirs[j]);
        std::printf("%s %s %zu %zu %lld\n", specs[j].raw.c_str(), dirs[j].c_str(),
                    m.featureCount(), m.unitScaleFeatures(),
                    static_cast<long long>(nan_filled[j]));
    }
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    std::string weights, regimes, perturb, tmp = "/tmp";
    bool self = false, no_scaler = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--selftest")) self = true;
        else if (!std::strcmp(argv[i], "--no-scaler")) no_scaler = true;
        else if (!std::strcmp(argv[i], "--weights") && i + 1 < argc) weights = argv[++i];
        else if (!std::strcmp(argv[i], "--regimes") && i + 1 < argc) regimes = argv[++i];
        else if (!std::strcmp(argv[i], "--perturb-coef") && i + 1 < argc) perturb = argv[++i];
        else if (!std::strcmp(argv[i], "--tmp") && i + 1 < argc) tmp = argv[++i];
    }
    try {
        if (self) return selftest(tmp);
        if (weights.empty() || regimes.empty()) {
            std::fprintf(stderr,
                         "usage: %s --selftest [--tmp DIR]\n"
                         "       %s --weights DIR --regimes <code[.code]*:L|S,...>\n"
                         "            [--no-scaler] [--perturb-coef R:F:DELTA]\n",
                         argv[0], argv[0]);
            return 2;
        }
        return run(weights, regimes, no_scaler, perturb);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "model_parity_driver: %s\n", e.what());
        return 1;
    }
}
