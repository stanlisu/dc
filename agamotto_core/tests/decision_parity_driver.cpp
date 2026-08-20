// PHASE 5 gate driver. Reads one symbol's raw OHLCV panel as CSV on stdin,
// engineers the 65-column panel, applies the REGIME GATE, scores each firing
// regime's model, and takes the DECISION — for EVERY row.
//
// WHY THE WHOLE CHAIN AND NOT SYNTHETIC y_pred. A driver handed made-up
// predictions would grade four comparison operators. The property under test is
// that the decision the LIVE core takes on a real panel is the decision the
// REFERENCE takes on the same panel, and the only way a `>` that should be a
// `<` shows up as a wrong SIDE is if the numbers reaching it are the real ones.
//
// WHY EVERY ROW. Live decides on one row per bar. Grading only the newest row
// would compare five decisions per scenario — and a decision is a BOOLEAN, so
// five samples is nowhere near enough to catch a rule that is right in the
// middle of the distribution and wrong at the edges. The rule is row-
// independent, so all 699 rows are graded and the newest one is reported
// separately.
//
//   --weights <dir>       the export_agamotto_sentinel_weights.py output dir
//   --regimes <spec>      comma-separated, each `code[.code]*:L|S`
//                         CODES ONLY, the same thing that crosses ICore.
//   --threshold-long      \
//   --threshold-short      |  the five algo_params keys, REQUIRED. There is no
//   --center-long          |  default for any of them: a defaulted width is the
//   --center-short         |  always-on gate and a defaulted centre silently
//   --reverse             /   reproduces the pre-2026-08-08 zero-centred one.
//
//   --print-floor         print kAbsThreshFloor and exit. The harness asserts it
//                         equals gauntlet.thresholds.ABS_THRESH_FLOOR, so the
//                         C++ constant is a GATED copy of one source rather than
//                         a second declaration of the same number.
//   --selftest            the refuse-to-load tests for the gate (below).
//
// Output:
//   #gate
//   <threshold_long threshold_short center_long center_short reverse
//    edge_long edge_short>
//   #panel
//   <65 column codes>
//   <PANEL_BARS rows, %.17g>
//   #preds
//   <the --regimes specs, echoed verbatim as the header>
//   <PANEL_BARS rows of %.17g; NaN where the regime gate did not fire>
//   #decisions
//   row fired side y_pred threshold threshold_center n_triggered n_long n_short
//       net_count winning_index
//   <PANEL_BARS rows>
//   #meta
//   <one line per regime: spec dir n_features>
#include "../src/decision_rule.hpp"
#include "../src/feature_engine.hpp"
#include "../src/model_runner.hpp"
#include "../src/regime_gate.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
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
        throw std::invalid_argument("decision_parity_driver: unparseable value '" + s + "'");
    return v;
}

struct ParsedSpec {
    std::vector<uint16_t> atoms;
    Position pos{Position::LONG};
    std::string raw;
};

// Identical parse to tests/model_parity_driver.cpp and
// tests/regime_parity_driver.cpp — deliberately duplicated rather than shared,
// for the reason stated there: a helper header between parity drivers is a
// place for one bug to make all of them agree.
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

RawBars readRawCsv(std::istream& in)
{
    std::string header;
    if (!std::getline(in, header))
        throw std::invalid_argument("decision_parity_driver: empty stdin");
    const std::vector<std::string> cols = split(header, ',');
    std::vector<std::vector<double>> data(cols.size());
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        const std::vector<std::string> f = split(line, ',');
        if (f.size() != cols.size())
            throw std::invalid_argument("decision_parity_driver: ragged row");
        for (size_t j = 0; j < f.size(); ++j) data[j].push_back(toDouble(f[j]));
    }
    RawBars rb;
    for (size_t j = 0; j < cols.size(); ++j) {
        const std::string& c = cols[j];
        if      (c == "open")   rb.open = data[j];
        else if (c == "high")   rb.high = data[j];
        else if (c == "low")    rb.low = data[j];
        else if (c == "close")  rb.close = data[j];
        else if (c == "volume") rb.volume = data[j];
        else if (c == "quote_volume") rb.quote_volume = data[j];
        else if (c == "taker_buy_quote_volume") rb.taker_buy_quote_volume = data[j];
        else if (c == "number_of_trades") rb.number_of_trades = data[j];
        else throw std::invalid_argument("decision_parity_driver: unknown column " + c);
    }
    return rb;
}

// The REFUSE-TO-LOAD tests. Every one of these is a gate that would otherwise
// run, look healthy and trade at a threshold nobody chose.
int selfTest()
{
    int failures = 0;
    auto expectThrow = [&](const char* what, GateParams g) {
        try {
            g.validate();
            std::printf("FAIL: %s was ACCEPTED\n", what);
            ++failures;
        } catch (const std::invalid_argument&) {
            std::printf("ok:   %s refused\n", what);
        }
    };

    GateParams good;
    good.threshold_long = 0.00070083;
    good.threshold_short = 0.0013514;
    good.threshold_center_long = -0.00012442;
    good.threshold_center_short = 0.00030606;
    good.reverse = 1;
    try {
        good.validate();
        std::printf("ok:   the DEPLOYED gate validates\n");
    } catch (const std::exception& e) {
        std::printf("FAIL: the deployed gate was refused: %s\n", e.what());
        ++failures;
    }

    // The floor, from both sides of it. 0.0002 exactly must PASS (a value that
    // IS the floor is legal); anything under must be refused, and refused
    // rather than raised to it.
    {
        GateParams g = good;
        g.threshold_long = 0.0002;
        g.threshold_short = 0.0002;
        try {
            g.validate();
            std::printf("ok:   |threshold| == the floor (%.17g) validates\n",
                        kAbsThreshFloor);
        } catch (const std::exception& e) {
            std::printf("FAIL: the floor value itself was refused: %s\n", e.what());
            ++failures;
        }
    }
    { GateParams g = good; g.threshold_long = 0.00019;
      expectThrow("threshold_long BELOW the 2 bps floor", g); }
    { GateParams g = good; g.threshold_short = 0.0;
      expectThrow("threshold_short == 0 (the always-on gate)", g); }
    { GateParams g = good; g.threshold_long = -0.0007;
      expectThrow("a NEGATIVE width (the sign belongs to the position)", g); }
    { GateParams g = good; g.threshold_short = std::numeric_limits<double>::quiet_NaN();
      expectThrow("a NaN width", g); }
    { GateParams g = good;
      g.threshold_center_long = std::numeric_limits<double>::infinity();
      expectThrow("an infinite centre", g); }
    { GateParams g = good; g.reverse = 0;
      expectThrow("reverse == 0 (a permanently flat bot)", g); }
    { GateParams g = good; g.reverse = 2;
      expectThrow("reverse == 2 (a silent doubling of live size)", g); }

    // A gate that clamped instead of refusing would still be running; prove the
    // accepted one is UNCHANGED, value for value.
    {
        const LegGate l = good.leg(Position::LONG);
        const LegGate s = good.leg(Position::SHORT);
        const double want_l = -0.00012442 + 0.00070083;
        const double want_s = 0.00030606 - 0.0013514;
        if (std::fabs(l.edge - want_l) > 1e-18 || std::fabs(s.edge - want_s) > 1e-18) {
            std::printf("FAIL: edges %.17g / %.17g, expected %.17g / %.17g\n",
                        l.edge, s.edge, want_l, want_s);
            ++failures;
        } else {
            std::printf("ok:   edges long=%.17g short=%.17g (== the DEPLOYED stack's "
                        "optimal_threshold column)\n", l.edge, s.edge);
        }
        // The long edge is POSITIVE here and the short edge NEGATIVE, but
        // nothing may infer a position from that: measured -0.000689 on another
        // arm's LONG leg. Asserted only that they are not equal, which is what
        // a collapsed per-leg gate would produce.
        if (l.edge == s.edge) {
            std::printf("FAIL: the two legs share one edge — the per-leg gate "
                        "collapsed\n");
            ++failures;
        }
    }

    // A non-finite y_pred must NEVER vote. `inf > edge` is true and
    // `-inf < edge` is true, so without the explicit finite check a poisoned
    // column votes on one leg or the other on every bar.
    {
        const LegGate l = good.leg(Position::LONG);
        const LegGate s = good.leg(Position::SHORT);
        const double inf = std::numeric_limits<double>::infinity();
        const double nan = std::numeric_limits<double>::quiet_NaN();
        const bool bad = legFires(inf, l, Position::LONG)
                      || legFires(-inf, s, Position::SHORT)
                      || legFires(nan, l, Position::LONG)
                      || legFires(nan, s, Position::SHORT);
        if (bad) { std::printf("FAIL: a non-finite y_pred VOTED\n"); ++failures; }
        else     { std::printf("ok:   inf / NaN never vote\n"); }
    }

    // Mismatched vector lengths must throw rather than silently pairing a
    // prediction with another regime's leg.
    {
        std::vector<Position> pos{Position::LONG, Position::SHORT};
        std::vector<double> y{0.01};
        try {
            evaluateDecision(good, pos, y);
            std::printf("FAIL: a length mismatch was ACCEPTED\n");
            ++failures;
        } catch (const std::invalid_argument&) {
            std::printf("ok:   a positions/predictions length mismatch throws\n");
        }
    }

    // ---- THE VOTE AND THE REPRESENTATIVE REGIME, on hand-built ballots ----
    //
    // These four cases exist because the SUITE cannot exercise them: across
    // 6990 real decisions only TWO rows have both legs voting at once, and on
    // neither does the minority leg hold the larger |y_pred|. A rule that
    // picked the representative regime over ALL voters instead of the majority
    // leg would therefore agree with the correct one on every row the parity
    // gate ever sees. Graded here instead, on ballots built to separate them.
    {
        const LegGate L = good.leg(Position::LONG);
        const LegGate S = good.leg(Position::SHORT);
        auto check = [&](const char* what, const DecisionOutcome& d, int side,
                         int n_trig, int n_long, int n_short, int win) {
            if (d.side == side && d.n_triggered == n_trig && d.n_long == n_long
                && d.n_short == n_short && d.winning_index == win
                && d.fired == (side != 0)) {
                std::printf("ok:   %s\n", what);
            } else {
                std::printf("FAIL: %s -> side=%d fired=%d n_trig=%d %dL/%dS win=%d "
                            "(wanted side=%d n_trig=%d %dL/%dS win=%d)\n",
                            what, d.side, d.fired ? 1 : 0, d.n_triggered, d.n_long,
                            d.n_short, d.winning_index, side, n_trig, n_long,
                            n_short, win);
                ++failures;
            }
        };

        // (a) TWO longs and one short. The SHORT holds by far the largest
        // |y_pred|, and the decision is LONG. The representative must be the
        // bigger of the two LONGS — index 1 — not the short.
        {
            std::vector<Position> pos{Position::LONG, Position::LONG, Position::SHORT};
            std::vector<double> y{L.edge + 1e-6, L.edge + 1e-4, S.edge - 1.0};
            const DecisionOutcome d = evaluateDecision(good, pos, y);
            check("majority-leg representative (short has the largest |y_pred| "
                  "but the decision is LONG)", d, +1, 3, 2, 1, 1);
            if (d.winning_index == 1
                && (d.threshold != L.width || d.threshold_center != L.center)) {
                std::printf("FAIL: the reported width/centre are not the LONG "
                            "leg's\n");
                ++failures;
            }
        }
        // (b) A TIE keeps the LOWEST stack index.
        {
            std::vector<Position> pos{Position::LONG, Position::LONG};
            std::vector<double> y{L.edge + 1e-4, L.edge + 1e-4};
            check("a tie in |y_pred| keeps the lowest stack index",
                  evaluateDecision(good, pos, y), +1, 2, 2, 0, 0);
        }
        // (c) ONE EACH nets to zero: FLAT, and both still counted as votes. A
        // rule that summed instead of netting would call this a 2-vote long.
        {
            std::vector<Position> pos{Position::LONG, Position::SHORT};
            std::vector<double> y{L.edge + 1e-4, S.edge - 1e-4};
            // With net == 0 there IS no majority leg, so the representative
            // falls back to all voters and index 1 (the short) wins on |y_pred|
            // — 0.00114534 against the long's 0.00067641. The decision is still
            // FLAT; the representative is only what the log line names.
            const DecisionOutcome d = evaluateDecision(good, pos, y);
            check("one vote each nets to FLAT (not a 2-vote long)",
                  d, 0, 2, 1, 1, 1);
        }
        // (d) REVERSE = -1 flips the SIDE and nothing else: the votes, the
        // net and the representative regime's leg are unchanged.
        {
            GateParams rev = good;
            rev.reverse = -1;
            std::vector<Position> pos{Position::LONG, Position::LONG};
            std::vector<double> y{L.edge + 1e-6, L.edge + 1e-4};
            const DecisionOutcome d = evaluateDecision(rev, pos, y);
            check("reverse=-1 turns a 2-vote LONG ballot into a SHORT decision",
                  d, -1, 2, 2, 0, 1);
            if (d.net_count != 2) {
                std::printf("FAIL: reverse changed net_count (%d), it must only "
                            "change the side\n", d.net_count);
                ++failures;
            }
        }
    }

    std::printf("%s\n", failures ? "=== SELFTEST FAILED ===" : "=== SELFTEST OK ===");
    return failures ? 1 : 0;
}

} // namespace

int main(int argc, char** argv)
{
    std::string weights, regimes;
    bool have_tl = false, have_ts = false, have_cl = false, have_cs = false, have_rv = false;
    GateParams gate;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto need = [&](const char* what) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "decision_parity_driver: %s needs a value\n", what);
                std::exit(2);
            }
            return argv[++i];
        };
        if      (a == "--selftest")    return selfTest();
        else if (a == "--print-floor") { std::printf("%.17g\n", kAbsThreshFloor); return 0; }
        else if (a == "--weights")     weights = need("--weights");
        else if (a == "--regimes")     regimes = need("--regimes");
        else if (a == "--threshold-long")  { gate.threshold_long = toDouble(need(a.c_str())); have_tl = true; }
        else if (a == "--threshold-short") { gate.threshold_short = toDouble(need(a.c_str())); have_ts = true; }
        else if (a == "--center-long")     { gate.threshold_center_long = toDouble(need(a.c_str())); have_cl = true; }
        else if (a == "--center-short")    { gate.threshold_center_short = toDouble(need(a.c_str())); have_cs = true; }
        else if (a == "--reverse")         { gate.reverse = std::atoi(need(a.c_str()).c_str()); have_rv = true; }
        else {
            std::fprintf(stderr, "decision_parity_driver: unknown arg %s\n", a.c_str());
            return 2;
        }
    }
    if (weights.empty() || regimes.empty()) {
        std::fprintf(stderr, "decision_parity_driver: --weights and --regimes are required\n");
        return 2;
    }
    // REQUIRED, all five. A driver that defaulted one would grade a gate the
    // harness never asked for and report it as parity.
    if (!have_tl || !have_ts || !have_cl || !have_cs || !have_rv) {
        std::fprintf(stderr,
            "decision_parity_driver: --threshold-long/--threshold-short/"
            "--center-long/--center-short/--reverse are ALL required; there is "
            "no default for any of them\n");
        return 2;
    }

    try {
        // THROWS on a sub-floor / negative / non-finite / bad-reverse gate,
        // BEFORE anything is engineered. This is the same call createCore makes.
        gate.validate();

        std::vector<ParsedSpec> specs;
        for (const std::string& tok : split(regimes, ',')) specs.push_back(parseSpec(tok));

        std::vector<std::string> dirs;
        dirs.reserve(specs.size());
        for (const ParsedSpec& sp : specs) dirs.push_back(regimeDirName(sp.atoms, sp.pos));

        ModelBook book;
        book.load(weights, dirs);

        const RawBars rb = readRawCsv(std::cin);
        Table panel = engineerFeatures(rb);
        ModelBook::assertPanelLayout(panel);
        const size_t rows = panel.cols.empty() ? 0 : panel.cols.front().size();

        const LegGate long_gate = gate.leg(Position::LONG);
        const LegGate short_gate = gate.leg(Position::SHORT);
        std::printf("#gate\n%.17g %.17g %.17g %.17g %d %.17g %.17g\n",
                    gate.threshold_long, gate.threshold_short,
                    gate.threshold_center_long, gate.threshold_center_short,
                    gate.reverse, long_gate.edge, short_gate.edge);

        std::printf("#panel\n");
        for (size_t j = 0; j < panel.size(); ++j)
            std::printf("%s%s", j ? "," : "", panel.names[j].c_str());
        std::printf("\n");
        for (size_t r = 0; r < rows; ++r) {
            for (size_t j = 0; j < panel.size(); ++j)
                std::printf("%s%.17g", j ? "," : "", panel.cols[j][r]);
            std::printf("\n");
        }

        // The gate masks, whole-panel — the same call the core makes, for the
        // same reason (several predicates are only meaningful panel-wide).
        std::vector<std::vector<char>> masks;
        std::vector<Position> positions;
        masks.reserve(specs.size());
        positions.reserve(specs.size());
        for (const ParsedSpec& sp : specs) {
            masks.push_back(regimeMask(panel, sp.atoms, sp.pos));
            positions.push_back(sp.pos);
        }

        // Predictions: NaN wherever the regime gate did not let the row through,
        // exactly as the core leaves them. The reference never predicts a
        // filtered-out row at all, so the two are the same statement.
        std::vector<std::vector<double>> preds(
            specs.size(), std::vector<double>(rows, std::numeric_limits<double>::quiet_NaN()));
        int64_t nan_filled = 0;
        for (size_t i = 0; i < specs.size(); ++i) {
            const LinearModel& m = book.at(dirs[i]);
            for (size_t r = 0; r < rows; ++r) {
                if (masks[i][r] == 0) continue;
                preds[i][r] = m.predictRow(panel, r, &nan_filled);
            }
        }

        std::printf("#preds\n%s\n", regimes.c_str());
        for (size_t r = 0; r < rows; ++r) {
            for (size_t i = 0; i < specs.size(); ++i)
                std::printf("%s%.17g", i ? "," : "", preds[i][r]);
            std::printf("\n");
        }

        std::printf("#decisions\n");
        std::printf("row,fired,side,y_pred,threshold,threshold_center,n_triggered,"
                    "n_long,n_short,net_count,winning_index\n");
        std::vector<double> row_preds(specs.size());
        for (size_t r = 0; r < rows; ++r) {
            for (size_t i = 0; i < specs.size(); ++i) row_preds[i] = preds[i][r];
            const DecisionOutcome d = evaluateDecision(gate, positions, row_preds);
            std::printf("%zu,%d,%d,%.17g,%.17g,%.17g,%d,%d,%d,%d,%d\n",
                        r, d.fired ? 1 : 0, d.side, d.y_pred, d.threshold,
                        d.threshold_center, d.n_triggered, d.n_long, d.n_short,
                        d.net_count, d.winning_index);
        }

        std::printf("#meta\n");
        for (size_t i = 0; i < specs.size(); ++i)
            std::printf("%s %s %zu\n", specs[i].raw.c_str(), dirs[i].c_str(),
                        book.at(dirs[i]).featureCount());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "decision_parity_driver: %s\n", e.what());
        return 1;
    }
}
