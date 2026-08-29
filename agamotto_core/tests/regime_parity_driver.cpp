// PHASE 3 gate driver. Reads one symbol's raw OHLCV panel as CSV on stdin,
// engineers the 65-column panel, evaluates the regime stack named on the
// command line, and prints BOTH on stdout.
//
// WHY BOTH. tests/regime_parity.py must drive the REAL reference
// (research_filters.apply_filter_mask) over the *SAME* engineered panel this
// binary gated — not over a panel Python engineered alongside. Feature parity
// is already gated separately at 1e-9 relative, and 1e-9 is enormous next to a
// boolean: a cell sitting 1e-12 away from `adx > 25` would flip the mask on one
// side and the run would go red for a reason that is not the gate. Emitting the
// panel makes the comparison EXACT and makes it about the predicates alone.
//
//   --regimes <spec>   comma-separated, each `code[.code]*:L|S`
//                      e.g. 60.75:L,29.66.73:L,69.65:S
//                      CODES ONLY — the same thing that crosses ICore. No
//                      regime name is accepted, parsed or printed here.
//   --selftest         the atomIsKnown/atomMask consistency sweep (below); no
//                      stdin, no panel.
//
// Output:
//   #panel
//   <65 column codes>
//   <PANEL_BARS rows, %.17g>
//   #masks
//   <the --regimes specs, echoed verbatim as the header>
//   <PANEL_BARS rows of 0/1>
//
// NaN and inf ride out of the panel untouched, exactly as
// tests/feature_parity_driver.cpp emits them: the harness reads them back with
// strtod and hands pandas the same values, which is what makes "x > NaN is
// False" testable rather than assumed.
#include "../src/feature_engine.hpp"
#include "../src/regime_gate.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
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
        throw std::invalid_argument("regime_parity_driver: unparseable value '" + s + "'");
    return v;
}

struct ParsedSpec {
    std::vector<uint16_t> atoms;
    Position pos{Position::LONG};
    std::string raw;
};

// `code[.code]*:L|S`. Strict: anything unparseable THROWS rather than being
// skipped. A silently dropped regime would shrink the comparison and still
// print PASS — the failure sentinel_core's own harness records (155 columns
// quietly becoming 55).
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

// The consistency sweep promised in regime_gate.hpp. atomIsKnown() is what
// setRegimeStack() validates against, and atomMask() is what runs every bar; if
// they disagree in either direction the core either accepts a stack it cannot
// evaluate (throws forever, 7.3 days after boot) or refuses one it could.
// Swept over the whole code space rather than over a list, so a code added to
// one and not the other is caught without anyone remembering to update a test.
int selftest()
{
    // A panel of the right WIDTH is needed because engineerFeatures pins it;
    // the VALUES are irrelevant here — the sweep asks only whether a predicate
    // exists, and a throw from a missing COLUMN would be a different failure.
    RawBars rb;
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
    const Table t = engineerFeatures(rb);

    int failures = 0;
    int known = 0;
    for (uint32_t c = 0; c < 4096; ++c) {
        const uint16_t code = static_cast<uint16_t>(c);
        const bool declared = atomIsKnown(code);
        bool evaluable = true;
        try {
            (void)atomMask(t, code, Position::LONG);
        } catch (const std::exception&) {
            evaluable = false;
        }
        if (declared != evaluable) {
            std::printf("  FAIL: code %u atomIsKnown=%d atomMask-evaluable=%d\n",
                        c, static_cast<int>(declared), static_cast<int>(evaluable));
            ++failures;
        }
        if (declared) ++known;
    }
    std::printf("  ok:   %d atom predicates, atomIsKnown agrees with atomMask on"
                " all 4096 codes\n", known);

    // An empty conjunction is the unconditional always-fire gate (`baseline`,
    // removed forever 2026-06-18). It must THROW, not return all-True.
    try {
        (void)regimeMask(t, {}, Position::LONG);
        std::printf("  FAIL: an EMPTY regime was evaluated instead of refused\n");
        ++failures;
    } catch (const std::exception&) {
        std::printf("  ok:   an EMPTY regime (the always-fire gate) is refused\n");
    }

    std::printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "PASSED",
                failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}

int run(const std::string& regimes_arg)
{
    std::vector<ParsedSpec> specs;
    for (const std::string& s : split(regimes_arg, ',')) {
        if (!s.empty()) specs.push_back(parseSpec(s));
    }
    if (specs.empty())
        throw std::invalid_argument("regime_parity_driver: --regimes named nothing");

    std::string header;
    if (!std::getline(std::cin, header))
        throw std::invalid_argument("regime_parity_driver: empty input, no header");
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
                "regime_parity_driver: row " + std::to_string(rows) + " has " +
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

    std::printf("#panel\n");
    for (size_t j = 0; j < t.names.size(); ++j)
        std::printf("%s%c", t.names[j].c_str(), j + 1 == t.names.size() ? '\n' : ',');
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < t.cols.size(); ++j)
            std::printf("%.17g%c", t.cols[j][i], j + 1 == t.cols.size() ? '\n' : ',');
    }

    std::vector<std::vector<char>> masks;
    masks.reserve(specs.size());
    for (const ParsedSpec& s : specs) masks.push_back(regimeMask(t, s.atoms, s.pos));

    std::printf("#masks\n");
    for (size_t j = 0; j < specs.size(); ++j)
        std::printf("%s%c", specs[j].raw.c_str(), j + 1 == specs.size() ? '\n' : ',');
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < masks.size(); ++j)
            std::printf("%d%c", static_cast<int>(masks[j][i]),
                        j + 1 == masks.size() ? '\n' : ',');
    }
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    std::string regimes;
    bool self = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--selftest")) self = true;
        else if (!std::strcmp(argv[i], "--regimes") && i + 1 < argc) regimes = argv[++i];
    }
    // Reported as a MESSAGE on stderr rather than left to terminate(): the
    // harness prints stderr, and an abort with empty stdout reads like "no
    // rows" instead of like a rejected spec.
    try {
        if (self) return selftest();
        if (regimes.empty()) {
            std::fprintf(stderr, "usage: %s --selftest | --regimes <code[.code]*:L|S,...>\n",
                         argv[0]);
            return 2;
        }
        return run(regimes);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "regime_parity_driver: %s\n", e.what());
        return 1;
    }
}
