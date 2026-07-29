// Replay events -> bars -> features -> regime masks. Prints one CSV column per
// "<regime>|<position>" so the Python reference can be diffed against it.
#include "../src/bar_builder.hpp"
#include "../src/feature_engine.hpp"
#include "../src/regime_gate.hpp"

#include <cstdio>
#include <iostream>
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

int main(int argc, char** argv)
{
    int bar_sec = 5, target_sec = 30;
    if (argc >= 2) bar_sec = std::stoi(argv[1]);
    if (argc >= 3) target_sec = std::stoi(argv[2]);
    // Remaining args: regime specs "name|position"
    std::vector<std::pair<std::string, std::string>> specs;
    for (int i = 3; i < argc; ++i) {
        auto p = split(argv[i], '|');
        specs.emplace_back(p[0], p.size() > 1 ? p[1] : "long");
    }

    BarBuilder bb(bar_sec, target_sec);
    std::vector<Bar> bars;
    std::string line;
    while (std::getline(std::cin, line)) {
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

    FeatureEngine fe({30, 60, 300, 900}, bar_sec, target_sec);
    std::vector<std::string> names;
    std::vector<std::vector<double>> cols;
    fe.compute(bars, names, cols);
    FeaturePanel panel(names, cols);

    std::vector<std::vector<char>> masks;
    for (const auto& s : specs) masks.push_back(applyFilterMask(panel, s.first, s.second));

    for (size_t j = 0; j < specs.size(); ++j)
        std::printf("%s|%s%c", specs[j].first.c_str(), specs[j].second.c_str(),
                    j + 1 == specs.size() ? '\n' : ',');
    for (size_t i = 0; i < bars.size(); ++i)
        for (size_t j = 0; j < masks.size(); ++j)
            std::printf("%d%c", masks[j][i] ? 1 : 0, j + 1 == masks.size() ? '\n' : ',');
    return 0;
}
