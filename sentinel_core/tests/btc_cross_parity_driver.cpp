// Anchor cross-feature parity: build TWO symbols from two event streams, treat
// the first as the anchor, and emit the PEER's panel including cross-features.
//
// The anchor slim frame is aligned to the peer's bars BY bar_ts — never by
// position and never "latest". A positional join looks right on dense data and
// silently corrupts the moment either symbol skips a bar.
#include "../src/bar_builder.hpp"
#include "../src/feature_engine.hpp"

#include <cstdio>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace mjolnir;

static std::vector<std::string> split(const std::string& s, char d)
{
    std::vector<std::string> o; std::stringstream ss(s); std::string it;
    while (std::getline(ss, it, d)) o.push_back(it);
    return o;
}

static std::vector<Bar> buildBars(const std::string& path, int bar_sec, int target_sec)
{
    BarBuilder bb(bar_sec, target_sec);
    std::vector<Bar> bars;
    std::ifstream fh(path);
    std::string line;
    while (std::getline(fh, line)) {
        if (line.empty()) continue;
        auto f = split(line, ',');
        const std::string& t = f[0];
        const int64_t ts = std::stoll(f[1]);
        if (t == "T") {
            Bar out;
            if (bb.onTrade(std::stod(f[2]), std::stod(f[3]), std::stoi(f[4]) != 0, ts,
                           std::stoll(f[5]), &out)) bars.push_back(out);
        } else if (t == "B") {
            bb.onBookTicker(std::stod(f[2]), std::stod(f[3]), std::stod(f[4]), std::stod(f[5]), ts);
        } else if (t == "D") {
            double bp[5], bq[5], ap[5], aq[5];
            for (int i = 0; i < 5; ++i) {
                bp[i] = std::stod(f[2 + i]);  bq[i] = std::stod(f[7 + i]);
                ap[i] = std::stod(f[12 + i]); aq[i] = std::stod(f[17 + i]);
            }
            bb.onDepth(bp, bq, ap, aq, 5, ts);
        } else if (t == "M") {
            bb.onMarkPrice(std::stod(f[2]), std::stod(f[3]), std::stod(f[4]), std::stod(f[5]), ts);
        } else if (t == "L") {
            bb.onLiquidation(std::stoi(f[2]) != 0, std::stod(f[3]), ts);
        } else if (t == "O") {
            bb.setOpenInterest(std::stod(f[2]), ts);
        }
    }
    return bars;
}

int main(int argc, char** argv)
{
    if (argc < 3) { std::fprintf(stderr, "usage: %s <anchor.csv> <peer.csv>\n", argv[0]); return 2; }
    const int bar_sec = 5, target_sec = 30;

    const auto abars = buildBars(argv[1], bar_sec, target_sec);
    const auto pbars = buildBars(argv[2], bar_sec, target_sec);

    FeatureEngine fe({30, 60, 300, 900}, bar_sec, target_sec);
    std::vector<std::string> an, pn;
    std::vector<std::vector<double>> ac, pc;
    fe.compute(abars, an, ac);
    fe.compute(pbars, pn, pc);

    // Index the anchor's rows by bar timestamp, then align to the peer's bars.
    std::map<int64_t, size_t> a_by_ts;
    for (size_t i = 0; i < abars.size(); ++i) a_by_ts[abars[i].bucket_ms] = i;

    const size_t n = pbars.size();
    std::vector<double> anchor(n * FeatureEngine::ANCHOR_COLS,
                               std::numeric_limits<double>::quiet_NaN());
    for (size_t i = 0; i < n; ++i) {
        auto it = a_by_ts.find(pbars[i].bucket_ms);
        if (it == a_by_ts.end()) continue;          // no anchor bar -> NaN row
        FeatureEngine::extractAnchorRow(an, ac, it->second,
                                        &anchor[i * FeatureEngine::ANCHOR_COLS]);
    }

    fe.addAnchorCrossFeatures(pn, pc, anchor);

    for (size_t j = 0; j < pn.size(); ++j)
        std::printf("%s%c", pn[j].c_str(), j + 1 == pn.size() ? '\n' : ',');
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < pc.size(); ++j)
            std::printf("%.17g%c", pc[j][i], j + 1 == pc.size() ? '\n' : ',');
    return 0;
}
