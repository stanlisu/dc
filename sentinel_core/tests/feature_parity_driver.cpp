// Replay events -> bars -> features, print the feature panel as CSV.
// The Python reference consumes the SAME bars and prints the same panel.
#include "../src/bar_builder.hpp"
#include "../src/feature_engine.hpp"

#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace mjolnir;

static std::vector<std::string> split(const std::string& s, char d)
{
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, d)) out.push_back(item);
    return out;
}

int main(int argc, char** argv)
{
    int bar_sec = 5, target_sec = 30;
    if (argc >= 2) bar_sec = std::stoi(argv[1]);
    if (argc >= 3) target_sec = std::stoi(argv[2]);

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
                           std::stoll(f[5]), &out))
                bars.push_back(out);
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

    for (size_t j = 0; j < names.size(); ++j)
        std::printf("%s%c", names[j].c_str(), j + 1 == names.size() ? '\n' : ',');
    for (size_t i = 0; i < bars.size(); ++i) {
        for (size_t j = 0; j < cols.size(); ++j)
            std::printf("%.17g%c", cols[j][i], j + 1 == cols.size() ? '\n' : ',');
    }
    return 0;
}
