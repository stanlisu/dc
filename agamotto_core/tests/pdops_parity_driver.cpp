// Differential parity: agamotto::pdops == pandas 2.3.3, cell by cell.
//
// Reads the two CSVs emitted by tests/pdops_golden.py. The GOLDEN HEADER IS
// THE SPEC: each column name encodes the call to make (`rollskew|ret|14|14`),
// so the two sides cannot drift — a spec this driver cannot parse is a hard
// failure, never a skipped column.
//
// THE GATE, and why it is what it is:
//
//   1. IDENTICAL NaN MASKS. Zero tolerance, every column, including the PROBE
//      ones. A NaN that should be a number (or the reverse) is not a rounding
//      difference: downstream it becomes a regime that fires when it must not,
//      because `x > NaN` is False and a warmup cutoff that leaks a real value
//      fires on ~30% of the first bars instead of 0% (research.py:57-60).
//   2. max rel diff <= 1e-12 on every gated column.
//   3. skew/kurt are gated on |a-b| <= 1e-12 * max(|b|, 1) instead. They are
//      DIMENSIONLESS statistics that legitimately pass through zero — the
//      rolling skew of a return series sits at ~2e-3 — so a pure relative
//      metric divides a 1e-14 absolute agreement by a near-zero denominator
//      and reports 1e-11. The absolute half of the gate is the meaningful
//      statement at their natural O(1) scale.
//   4. NEG_ columns hold a DELIBERATELY WRONG computation. The gate is
//      INVERTED: the C++ must NOT match them. Without this, an implementation
//      that used population moments or a nearest-rank quantile would pass a
//      positive-only harness wherever the two happen to be close.
//   5. PROBE_ columns are reported, not gated — ill-conditioned inputs the
//      reference never sends to these primitives, where pandas cannot
//      reproduce itself either (pdops_golden.py prints the measurement).
#include "../src/feature_engine.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace agamotto;

namespace {

constexpr double NA = std::numeric_limits<double>::quiet_NaN();
int g_failures = 0;

std::vector<std::string> split(const std::string& s, char d)
{
    std::vector<std::string> out;
    std::string item;
    std::stringstream ss(s);
    while (std::getline(ss, item, d)) out.push_back(item);
    return out;
}

// strtod, never a stream: it is exactly-rounded, so a %.17g round trip is
// lossless. (pandas' own read_csv fast parser is NOT, which is worth knowing
// if anyone re-reads these CSVs from Python — pass float_precision="round_trip".)
double parse(const std::string& s)
{
    if (s == "nan" || s == "NaN" || s.empty()) return NA;
    return std::strtod(s.c_str(), nullptr);
}

struct Csv {
    std::vector<std::string> header;
    std::vector<std::vector<double>> cols;   // column-major
};

Csv readCsv(const char* path)
{
    std::ifstream fh(path);
    if (!fh) {
        std::fprintf(stderr, "FAIL: cannot open %s\n", path);
        std::exit(2);
    }
    Csv c;
    std::string line;
    if (!std::getline(fh, line)) {
        std::fprintf(stderr, "FAIL: %s is empty\n", path);
        std::exit(2);
    }
    c.header = split(line, ',');
    c.cols.resize(c.header.size());
    size_t row = 0;
    while (std::getline(fh, line)) {
        if (line.empty()) continue;
        auto f = split(line, ',');
        if (f.size() != c.header.size()) {
            std::fprintf(stderr, "FAIL: %s row %zu has %zu fields, header has %zu\n",
                         path, row, f.size(), c.header.size());
            std::exit(2);
        }
        for (size_t j = 0; j < f.size(); ++j) c.cols[j].push_back(parse(f[j]));
        ++row;
    }
    return c;
}

// ---------------------------------------------------------------------------
// spec dispatch
// ---------------------------------------------------------------------------
const std::map<std::string, std::vector<double>>* g_in = nullptr;

const std::vector<double>& series(const std::string& name)
{
    auto it = g_in->find(name);
    if (it == g_in->end()) {
        std::fprintf(stderr, "FAIL: spec names input series %s, which the input "
                             "CSV does not have\n", name.c_str());
        std::exit(2);
    }
    return it->second;
}

std::vector<double> compute(const std::string& spec)
{
    auto f = split(spec, '|');
    std::string op = f[0];
    // PROBE_/NEG_ change how the column is GRADED, never what is computed:
    // for a NEG_ column the C++ still computes the CORRECT primitive, which
    // is the whole point of the inverted assertion.
    if (op.rfind("PROBE_", 0) == 0) op = op.substr(6);
    if (op.rfind("NEG_", 0) == 0) {
        const std::string kind = op.substr(4);
        if (kind == "popskew") op = "rollskew";
        else if (kind == "popkurt") op = "rollkurt";
        else if (kind == "nearestquantile") op = "rollquantile";
        else {
            std::fprintf(stderr, "FAIL: unknown NEG_ kind %s\n", kind.c_str());
            std::exit(2);
        }
    }

    if (op == "diff") return pdops::diff(series(f[1]));
    if (op == "diffn") return pdops::diffN(series(f[1]), std::stoi(f[2]));
    if (op == "shift") return pdops::shift(series(f[1]), std::stoi(f[2]));
    if (op == "pctchange") return pdops::pctChange(series(f[1]), std::stoi(f[2]));
    if (op == "rollsum") return pdops::rollSum(series(f[1]), std::stoi(f[2]), std::stoi(f[3]));
    if (op == "rollmean") return pdops::rollMean(series(f[1]), std::stoi(f[2]), std::stoi(f[3]));
    if (op == "rollstd") return pdops::rollStd(series(f[1]), std::stoi(f[2]), std::stoi(f[3]));
    if (op == "rollskew") return pdops::rollSkew(series(f[1]), std::stoi(f[2]), std::stoi(f[3]));
    if (op == "rollkurt") return pdops::rollKurt(series(f[1]), std::stoi(f[2]), std::stoi(f[3]));
    if (op == "rollcorr")
        return pdops::rollCorr(series(f[1]), series(f[2]), std::stoi(f[3]), std::stoi(f[4]));
    if (op == "rollquantile")
        return pdops::rollQuantile(series(f[1]), std::stoi(f[2]), std::stoi(f[3]),
                                   std::strtod(f[4].c_str(), nullptr));
    if (op == "rollquantiles") {
        std::vector<double> qs;
        for (const std::string& q : split(f[4], ';')) qs.push_back(std::strtod(q.c_str(), nullptr));
        const size_t which = static_cast<size_t>(std::stoi(f[5]));
        auto all = pdops::rollQuantiles(series(f[1]), std::stoi(f[2]), std::stoi(f[3]), qs);
        if (which >= all.size()) {
            std::fprintf(stderr, "FAIL: spec %s asks for level %zu of %zu\n", spec.c_str(),
                         which, all.size());
            std::exit(2);
        }
        return all[which];
    }
    std::fprintf(stderr, "FAIL: unknown spec op '%s' in '%s'\n", op.c_str(), spec.c_str());
    std::exit(2);
}

// ---------------------------------------------------------------------------
// grading
// ---------------------------------------------------------------------------
struct Diff {
    int maskDiff = 0;       // NaN on one side only
    int infDiff = 0;        // infinite on one side only, or opposite signs
    double maxRel = 0.0;
    double maxAbs = 0.0;
    size_t worstRow = 0;
    int cells = 0;
    int differing = 0;      // cells differing by more than 1e-9 relative
};

Diff grade(const std::vector<double>& a, const std::vector<double>& b)
{
    Diff d;
    for (size_t i = 0; i < b.size(); ++i) {
        const double x = a[i], y = b[i];
        const bool nx = std::isnan(x), ny = std::isnan(y);
        if (nx != ny) { ++d.maskDiff; ++d.differing; continue; }
        if (nx) continue;
        if (std::isinf(x) || std::isinf(y)) {
            if (!(std::isinf(x) && std::isinf(y) && std::signbit(x) == std::signbit(y))) {
                ++d.infDiff;
                ++d.differing;
            }
            continue;
        }
        ++d.cells;
        const double ad = std::fabs(x - y);
        const double rel = ad / std::max(std::fabs(y), 1e-300);
        if (ad > d.maxAbs) d.maxAbs = ad;
        if (rel > d.maxRel) { d.maxRel = rel; d.worstRow = i; }
        if (rel > 1e-9) ++d.differing;
    }
    return d;
}

void report(const char* verdict, const std::string& spec, const Diff& d, const char* note)
{
    std::printf("%-4s %-44s mask=%-4d inf=%-3d maxrel=%9.3e maxabs=%9.3e %s\n",
                verdict, spec.c_str(), d.maskDiff, d.infDiff, d.maxRel, d.maxAbs, note);
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::printf("usage: %s <pdops_input.csv> <pdops_golden.csv>\n", argv[0]);
        return 2;
    }
    const Csv in = readCsv(argv[1]);
    const Csv gold = readCsv(argv[2]);

    std::map<std::string, std::vector<double>> inputs;
    for (size_t j = 0; j < in.header.size(); ++j) inputs[in.header[j]] = in.cols[j];
    g_in = &inputs;

    if (!gold.cols.empty() && gold.cols[0].size() != in.cols[0].size()) {
        std::fprintf(stderr, "FAIL: input has %zu rows, golden has %zu\n",
                     in.cols[0].size(), gold.cols[0].size());
        return 2;
    }
    std::printf("rows=%zu inputs=%zu specs=%zu\n\n", in.cols[0].size(), in.header.size(),
                gold.header.size());

    int gated = 0, probes = 0, negs = 0;

    std::printf("--- GATED: identical NaN masks AND max rel diff <= 1e-12 ---\n");
    for (size_t j = 0; j < gold.header.size(); ++j) {
        const std::string& spec = gold.header[j];
        if (spec.rfind("PROBE_", 0) == 0 || spec.rfind("NEG_", 0) == 0) continue;
        ++gated;
        const std::vector<double> got = compute(spec);
        const Diff d = grade(got, gold.cols[j]);
        const bool isMoment = spec.rfind("rollskew", 0) == 0 || spec.rfind("rollkurt", 0) == 0;
        // See the header comment: skew/kurt are graded on the mixed metric.
        bool ok = d.maskDiff == 0 && d.infDiff == 0;
        if (ok) {
            if (isMoment) {
                for (size_t i = 0; i < gold.cols[j].size() && ok; ++i) {
                    const double y = gold.cols[j][i], x = got[i];
                    if (std::isnan(y) || std::isinf(y)) continue;
                    if (std::fabs(x - y) > 1e-12 * std::max(std::fabs(y), 1.0)) ok = false;
                }
            } else {
                ok = d.maxRel <= 1e-12;
            }
        }
        if (!ok) ++g_failures;
        report(ok ? "ok" : "FAIL", spec, d, isMoment ? "[abs-or-rel 1e-12]" : "");
        if (!ok)
            std::printf("       worst row %zu: cpp=%.17g pandas=%.17g\n", d.worstRow,
                        got[d.worstRow], gold.cols[j][d.worstRow]);
    }

    std::printf("\n--- NEGATIVE: the C++ must NOT match these wrong computations ---\n");
    for (size_t j = 0; j < gold.header.size(); ++j) {
        const std::string& spec = gold.header[j];
        if (spec.rfind("NEG_", 0) != 0) continue;
        ++negs;
        const std::vector<double> got = compute(spec);
        const Diff d = grade(got, gold.cols[j]);
        const int total = d.cells + d.maskDiff + d.infDiff;
        // "Differs" means materially: more than a tenth of the compared cells
        // must be off by more than 1e-9 relative. A handful of differing cells
        // could be luck; a tenth of them cannot.
        const bool differs = total > 0 && d.differing * 10 > total;
        if (!differs) ++g_failures;
        std::printf("%-4s %-44s differing=%d/%d (%.1f%%) maxrel=%.3e\n",
                    differs ? "ok" : "FAIL", spec.c_str(), d.differing, total,
                    total ? 100.0 * d.differing / total : 0.0, d.maxRel);
        if (!differs)
            std::printf("       the C++ MATCHES a deliberately wrong reference — the "
                        "bias correction / interpolation is missing\n");
    }

    std::printf("\n--- PROBE: reported, NOT gated (ill-conditioned; see pdops_golden.py) ---\n");
    for (size_t j = 0; j < gold.header.size(); ++j) {
        const std::string& spec = gold.header[j];
        if (spec.rfind("PROBE_", 0) != 0) continue;
        ++probes;
        const std::vector<double> got = compute(spec);
        const Diff d = grade(got, gold.cols[j]);
        // The NaN mask is gated even here: a mask difference is never a
        // conditioning artifact.
        if (d.maskDiff != 0) {
            ++g_failures;
            report("FAIL", spec, d, "[mask is gated even for probes]");
        } else {
            report("--", spec, d, "");
        }
    }

    std::printf("\ngated=%d negative=%d probe=%d\n", gated, negs, probes);
    std::printf("%s (%d failure%s)\n", g_failures ? "FAILED" : "PASSED", g_failures,
                g_failures == 1 ? "" : "s");
    return g_failures ? 1 : 0;
}
