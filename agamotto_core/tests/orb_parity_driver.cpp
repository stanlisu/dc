// ORB context-timeframe parity -- the C++ half. See tests/orb_context_parity.py.
//
// For every base bar T in [start, end) and every symbol, decide exactly what the
// shipped core decides, out of the pieces the core is made of:
//
//   base panel      the 799 closed base bars ending AT T            engineerFeatures
//   context panel   per context timeframe, the newest 799 bars
//                   closed by D = T + base (what a REST fetch at D
//                   returns once the open bar is dropped)           engineerFeaturesContext
//   context row     contextAsofRow(close_ms, T, tf)                 context_asof.hpp
//   regime          positionAllowed(all atoms) AND every atom's
//                   atomMask at its own (panel, row)                regime_gate
//   y               the regime's exported linear model on the
//                   base panel's newest row                         model_runner
//   votes           the centred per-side gate                       evaluateDecision
//
// Regime names are parsed by SENTINEL's own parser (regime_name.hpp) -- the code
// that reads the stack live -- and the stack ORDER follows orb's
// `_load_regime_stack` (rows with an empty model or directory are skipped), so the
// regime index is the reference's index.
//
//   orb_parity_driver --klines DIR --symbols A,B --base 15m --context 1h,4h,1d
//       --stack CSV --weights DIR --gate CL,TL,CS,TS --start MS --end MS
//       --out cpp.jsonl [--threads N]
#include "agamotto_core.hpp"
#include "backfill_csv.hpp"
#include "context_asof.hpp"
#include "decision_rule.hpp"
#include "feature_engine.hpp"
#include "model_runner.hpp"
#include "regime_gate.hpp"
#include "regime_name.hpp"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace agamotto;

namespace {

std::vector<std::string> split(const std::string& s, char sep)
{
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string x;
    while (std::getline(ss, x, sep)) if (!x.empty()) out.push_back(x);
    return out;
}

int tfSec(const std::string& tf)
{
    const uint32_t v = regime_name::parseTfLabel(tf);
    if (v == 0) throw std::invalid_argument("not a timeframe: " + tf);
    return static_cast<int>(v);
}

std::vector<KlineBar> readCsv(const std::string& path, int tf_sec)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::string buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::vector<KlineBar> bars;
    backfill::ParseError err{};
    if (!backfill::parseRows(buf.data(), buf.size(), tf_sec, bars, err))
        throw std::runtime_error("cannot parse " + path + " line " + std::to_string(err.line));
    return bars;
}

RawBars raw(const std::vector<KlineBar>& v, size_t lo, size_t hi)
{
    RawBars rb;
    for (size_t i = lo; i < hi; ++i) {
        const KlineBar& b = v[i];
        rb.open.push_back(b.open);
        rb.high.push_back(b.high);
        rb.low.push_back(b.low);
        rb.close.push_back(b.close);
        rb.volume.push_back(b.volume);
        rb.quote_volume.push_back(b.quote_volume);
        rb.taker_buy_quote_volume.push_back(b.taker_buy_quote_volume);
        rb.number_of_trades.push_back(static_cast<double>(b.number_of_trades));
    }
    return rb;
}

struct Leg {
    std::vector<uint16_t> codes;
    std::vector<uint32_t> tfs;
    Position pos{Position::LONG};
    std::string dir;
};

// orb's _load_regime_stack: header-located columns, rows lacking a model or a
// directory skipped (the __summary__ row is one of them).
std::vector<Leg> readStack(const std::string& path)
{
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::string line;
    std::getline(f, line);
    // A CRLF file (python's csv.writer default) leaves '\r' on the last field.
    const auto chomp = [](std::string& x) { if (!x.empty() && x.back() == '\r') x.pop_back(); };
    chomp(line);
    const std::vector<std::string> hdr = split(line + ",", ',');
    int iR = -1, iP = -1, iM = -1, iD = -1;
    {
        std::stringstream hs(line);
        std::string c;
        int k = 0;
        while (std::getline(hs, c, ',')) {
            if (c == "regime") iR = k;
            if (c == "position") iP = k;
            if (c == "model") iM = k;
            if (c == "directory") iD = k;
            ++k;
        }
    }
    if (iR < 0 || iP < 0 || iM < 0 || iD < 0) throw std::runtime_error(path + ": header");
    std::vector<Leg> legs;
    while (std::getline(f, line)) {
        chomp(line);
        std::vector<std::string> c;
        std::stringstream ls(line);
        std::string x;
        while (std::getline(ls, x, ',')) c.push_back(x);
        while (c.size() < hdr.size()) c.push_back("");
        if (c[static_cast<size_t>(iM)].empty() || c[static_cast<size_t>(iD)].empty()) continue;
        RegimeSpec sp{};
        std::string why;
        if (!regime_name::parse(c[static_cast<size_t>(iR)], c[static_cast<size_t>(iP)], sp, why))
            throw std::runtime_error(c[static_cast<size_t>(iR)] + ": " + why);
        Leg l;
        for (uint8_t k = 0; k < sp.n_atoms; ++k) {
            l.codes.push_back(sp.atom_codes[k]);
            l.tfs.push_back(sp.atom_tf_sec[k]);
        }
        l.pos = sp.position > 0 ? Position::LONG : Position::SHORT;
        l.dir = regimeDirName(l.codes, l.tfs, l.pos);
        legs.push_back(l);
    }
    return legs;
}

struct CtxPanel {
    int64_t newest_open{-1};
    Table panel;
    std::vector<int64_t> close_ms;
};

std::string fmt(double y)
{
    char b[64];
    std::snprintf(b, sizeof(b), "%.17g", y);
    return b;
}

} // namespace

int main(int argc, char** argv)
{
    std::map<std::string, std::string> a;
    for (int i = 1; i + 1 < argc; i += 2) a[argv[i]] = argv[i + 1];
    for (const char* k : {"--klines", "--symbols", "--base", "--context", "--stack", "--weights",
                          "--gate", "--start", "--end", "--out"}) {
        if (!a.count(k)) { std::fprintf(stderr, "missing %s\n", k); return 2; }
    }
    const int threads = a.count("--threads") ? std::atoi(a["--threads"].c_str()) : 1;
    const int base_sec = tfSec(a["--base"]);
    const int64_t base_ms = static_cast<int64_t>(base_sec) * 1000;
    std::vector<int> ctx;
    for (const std::string& t : split(a["--context"], ',')) ctx.push_back(tfSec(t));
    const std::vector<std::string> syms = split(a["--symbols"], ',');
    const std::vector<std::string> g = split(a["--gate"], ',');
    if (g.size() != 4) { std::fprintf(stderr, "--gate CL,TL,CS,TS\n"); return 2; }
    GateParams gate;
    gate.threshold_center_long = std::strtod(g[0].c_str(), nullptr);
    gate.threshold_long = std::strtod(g[1].c_str(), nullptr);
    gate.threshold_center_short = std::strtod(g[2].c_str(), nullptr);
    gate.threshold_short = std::strtod(g[3].c_str(), nullptr);
    gate.reverse = 1;
    gate.validate();
    const int64_t start = std::strtoll(a["--start"].c_str(), nullptr, 10);
    const int64_t end = std::strtoll(a["--end"].c_str(), nullptr, 10);

    const std::vector<Leg> legs = readStack(a["--stack"]);
    std::vector<std::string> dirs;
    std::vector<Position> positions;
    for (const Leg& l : legs) {
        dirs.push_back(l.dir);
        positions.push_back(l.pos);
        for (const uint32_t tf : l.tfs) {
            const int t = static_cast<int>(tf);
            if (t != 0 && t != base_sec &&
                std::find(ctx.begin(), ctx.end(), t) == ctx.end())
                throw std::runtime_error(l.dir + " names a timeframe not in --context");
        }
    }
    ModelBook book;
    book.load(a["--weights"], dirs);
    std::fprintf(stderr, "stack: %zu regimes, models: %s\n", legs.size(), book.inventory().c_str());

    std::mutex out_mx;
    std::ofstream out(a["--out"]);
    std::atomic<size_t> next{0};
    std::atomic<int> errors{0};

    auto work = [&]() {
        for (;;) {
            const size_t si = next.fetch_add(1);
            if (si >= syms.size()) return;
            const std::string& sym = syms[si];
            try {
                const std::string kd = a["--klines"];
                const std::vector<KlineBar> base =
                    readCsv(kd + "/backfill_" + sym + "_" + a["--base"] + ".csv", base_sec);
                std::map<int, std::vector<KlineBar>> cbars;
                for (const std::string& t : split(a["--context"], ','))
                    cbars[tfSec(t)] = readCsv(kd + "/backfill_" + sym + "_" + t + ".csv", tfSec(t));
                std::map<int, CtxPanel> cache;
                std::string lines;
                for (size_t bi = 0; bi < base.size(); ++bi) {
                    const int64_t T = base[bi].bucket_open_ms;
                    if (T < start || T >= end) continue;
                    if (bi + 1 < PANEL_BARS)
                        throw std::runtime_error(sym + ": fewer than 799 base bars before T");
                    const Table bp = engineerFeatures(raw(base, bi + 1 - PANEL_BARS, bi + 1));
                    ModelBook::assertPanelLayout(bp);
                    const int64_t D = T + base_ms;
                    std::map<int, int> row;
                    for (const int tf : ctx) {
                        const int64_t tf_ms = static_cast<int64_t>(tf) * 1000;
                        const std::vector<KlineBar>& v = cbars[tf];
                        size_t hi = 0;   // bars closed by D: open + tf <= D
                        while (hi < v.size() && v[hi].bucket_open_ms + tf_ms <= D) ++hi;
                        if (hi == 0) { row[tf] = -1; continue; }
                        CtxPanel& cp = cache[tf];
                        if (cp.newest_open != v[hi - 1].bucket_open_ms) {
                            const size_t lo = hi > PANEL_BARS ? hi - PANEL_BARS : 0;
                            cp.newest_open = v[hi - 1].bucket_open_ms;
                            cp.panel = engineerFeaturesContext(raw(v, lo, hi));
                            cp.close_ms.clear();
                            for (size_t i = lo; i < hi; ++i)
                                cp.close_ms.push_back(v[i].bucket_open_ms + tf_ms);
                        }
                        AsofStatus st{};
                        row[tf] = contextAsofRow(cp.close_ms, T, tf_ms, &st, nullptr);
                    }
                    std::vector<double> y(legs.size(), std::nan(""));
                    for (size_t i = 0; i < legs.size(); ++i) {
                        const Leg& l = legs[i];
                        bool fired = positionAllowed(l.codes, l.pos);
                        for (size_t k = 0; k < l.codes.size(); ++k) {
                            const int tf = static_cast<int>(l.tfs[k]);
                            bool hit;
                            if (tf == 0 || tf == base_sec) {
                                hit = atomMask(bp, l.codes[k], l.pos).back() != 0;
                            } else if (row[tf] < 0) {
                                hit = false;
                            } else {
                                hit = atomMask(cache[tf].panel, l.codes[k], l.pos)
                                          .at(static_cast<size_t>(row[tf])) != 0;
                            }
                            fired = fired && hit;
                        }
                        if (fired) {
                            int64_t nan_filled = 0;
                            y[i] = book.at(l.dir).predictRow(
                                bp, bp.cols.front().size() - 1, &nan_filled);
                        }
                        lines += "{\"T\": " + std::to_string(T) + ", \"symbol\": \"" + sym +
                                 "\", \"regime\": " + std::to_string(i) + ", \"fired\": " +
                                 (fired ? "true" : "false") + ", \"y\": " +
                                 (fired ? fmt(y[i]) : std::string("null")) + "}\n";
                    }
                    const DecisionOutcome d = evaluateDecision(gate, positions, y);
                    lines += "{\"T\": " + std::to_string(T) + ", \"symbol\": \"" + sym +
                             "\", \"decision\": 1, \"long\": " + std::to_string(d.n_long) +
                             ", \"short\": " + std::to_string(d.n_short) + "}\n";
                }
                std::lock_guard<std::mutex> lk(out_mx);
                out << lines;
                std::fprintf(stderr, "  %s done\n", sym.c_str());
            } catch (const std::exception& e) {
                std::fprintf(stderr, "  %s FAILED: %s\n", sym.c_str(), e.what());
                ++errors;
            }
        }
    };
    std::vector<std::thread> pool;
    for (int t = 0; t < (threads < 1 ? 1 : threads); ++t) pool.emplace_back(work);
    for (std::thread& t : pool) t.join();
    return errors == 0 ? 0 : 1;
}
