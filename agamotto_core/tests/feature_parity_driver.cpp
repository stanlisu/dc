// Read one symbol's raw OHLCV panel as CSV on stdin, run agamotto's stage-2.2
// feature blocks AND the stage-2.3 TA-Lib indicator block over it, print the
// engineered panel as CSV on stdout.
//
// Links ta-lib (pinned 0.6.4) via src/talib_block.cpp — see
// tests/run_feature_parity.sh for the host recipe and the version check.
//
// The Python side (tests/feature_parity.py) generates the SAME CSV, feeds it to
// the REAL reference (`AgamottoResearch.engineer_features`), and diffs cell by
// cell. Nothing about the reference is reimplemented on either side of the
// seam: this binary computes, the harness compares.
//
// Values are printed with %.17g, which round-trips a double exactly; NaN and
// +/-inf print as "nan"/"inf"/"-inf" and Python's float() reads all three back.
// They are NOT sanitised on the way out — reproducing pandas' NaN and inf masks
// IS the contract (see src/feature_engine.hpp).
//
// Input header must name the columns; order is free and the three optional
// columns may be omitted entirely, which is how the "feed does not carry it"
// path gets exercised:
//     open,high,low,close,volume[,quote_volume][,taker_buy_quote_volume]
//     [,number_of_trades]
#include "../src/feature_engine.hpp"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace agamotto;

static std::vector<std::string> split(const std::string& s, char d)
{
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, d)) out.push_back(item);
    return out;
}

// strtod, not stod: the CSV is written with %.17g and must be read back
// exactly, and strtod is the same exactly-rounded conversion the pdops driver
// uses for its golden. It also accepts "nan"/"inf"/"-inf" without throwing.
static double toDouble(const std::string& s)
{
    const char* p = s.c_str();
    char* end = nullptr;
    const double v = std::strtod(p, &end);
    if (end == p)
        throw std::invalid_argument("feature_parity_driver: unparseable value '" + s + "'");
    return v;
}

static int run()
{
    std::string header;
    if (!std::getline(std::cin, header))
        throw std::invalid_argument("feature_parity_driver: empty input, no header");
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
                "feature_parity_driver: row " + std::to_string(rows) + " has " +
                std::to_string(f.size()) + " fields, header has " +
                std::to_string(cols.size()));
        for (size_t j = 0; j < cols.size(); ++j) raw[cols[j]].push_back(toDouble(f[j]));
        ++rows;
    }

    // An absent optional column stays an EMPTY vector, which is what
    // engineerFeatures reads as "the feed does not carry it". A missing
    // REQUIRED column arrives as empty too and is rejected there by width.
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

    for (size_t j = 0; j < t.names.size(); ++j)
        std::printf("%s%c", t.names[j].c_str(), j + 1 == t.names.size() ? '\n' : ',');
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < t.cols.size(); ++j)
            std::printf("%.17g%c", t.cols[j][i], j + 1 == t.cols.size() ? '\n' : ',');
    }
    return 0;
}

int main()
{
    // Caught and reported, not left to terminate(): the harness reads stderr
    // and prints it, so a width or column rejection must arrive as a MESSAGE
    // rather than as an abort with an empty stdout that reads like "no rows".
    try {
        return run();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "feature_parity_driver: %s\n", e.what());
        return 1;
    }
}
