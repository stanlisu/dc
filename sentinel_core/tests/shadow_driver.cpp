// REPLAY SHADOW — exercises the LIVE path through the public ICore interface.
//
// This is the half of M1 the variant-A gate cannot reach. m1_gate scores bars
// directly through the modules; this drives the contract the strategy actually
// uses at runtime:
//     makeCore() -> onTick() -> barReady() -> isWarm() -> decide() -> hold-TTL
//
// It is a REPLAY, not a live-feed run: the LTP feed/OMS are a separate
// operational concern. What it proves is that the wiring in core_impl works —
// bars accumulate, the warmup gate releases, decisions are produced, and the
// hold-TTL behaves — none of which the numerical gate touches.
//
// Usage: shadow_driver <algo_params.json> <events.csv>
#include "mjolnir_core.hpp"

#include <cstdio>
#include <fstream>
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
    if (argc < 3) { std::fprintf(stderr, "usage: %s <algo_params.json> <events.csv>\n", argv[0]); return 2; }

    std::printf("[shadow] core=%s real_impl=%d\n", coreBuildTag(),
                static_cast<int>(coreIsRealImplementation()));

    auto core = makeCore(argv[1]);
    if (!core) { std::fprintf(stderr, "makeCore returned null\n"); return 2; }
    std::printf("[shadow] core constructed, stack loaded\n");

    std::ifstream fh(argv[2]);
    if (!fh) { std::fprintf(stderr, "cannot open events\n"); return 2; }

    size_t ticks = 0, bars = 0, warm_bars = 0, fired = 0, held = 0;
    bool warm_announced = false;
    std::string line;

    while (std::getline(fh, line)) {
        if (line.empty()) continue;
        const auto f = split(line, ',');
        const std::string& t = f[0];
        const int64_t ts_ms = std::stoll(f[1]);

        TickEvent ev;
        ev.product_id = 1;
        ev.exchange_ts_ns = static_cast<uint64_t>(ts_ms) * 1000000ULL;

        if (t == "T") {
            ev.has_trade = true;
            ev.last_px = std::stod(f[2]);
            ev.last_qty = std::stod(f[3]);
            ev.aggressor_is_buy = (std::stoi(f[4]) != 0) ? 0 : 1;   // is_buyer_maker -> aggressor
        } else if (t == "B") {
            ev.has_book = true; ev.n_levels = 1;
            ev.bid_px[0] = std::stod(f[2]); ev.bid_qty[0] = std::stod(f[3]);
            ev.ask_px[0] = std::stod(f[4]); ev.ask_qty[0] = std::stod(f[5]);
        } else if (t == "D") {
            ev.has_book = true; ev.n_levels = 5;
            for (int i = 0; i < 5; ++i) {
                ev.bid_px[i] = std::stod(f[2 + i]);  ev.bid_qty[i] = std::stod(f[7 + i]);
                ev.ask_px[i] = std::stod(f[12 + i]); ev.ask_qty[i] = std::stod(f[17 + i]);
            }
        } else if (t == "M") {
            ev.mark_px = std::stod(f[2]); ev.index_px = std::stod(f[3]);
            ev.funding_rate = std::stod(f[4]);
        } else if (t == "L") {
            ev.liq_is_long = std::stoi(f[2]); ev.liq_notional = std::stod(f[3]);
        } else if (t == "O") {
            ev.open_interest = std::stod(f[2]);
        } else {
            continue;
        }

        core->onTick(ev);
        ++ticks;

        int64_t bar_ts = 0;
        if (core->barReady(&bar_ts)) {
            ++bars;
            if (!core->isWarm()) continue;
            if (!warm_announced) {
                std::printf("[shadow] WARM after %zu bars (bars_fed=%d)\n", bars,
                            core->barsBuffered());
                warm_announced = true;
            }
            ++warm_bars;
            const Decision d = core->decide();
            if (d.fired) {
                ++fired;
                if (fired <= 5 || fired % 50 == 0) {
                    std::printf("[MJDEC] bar_ts=%lld side=%d y_pred=%.10g thr=%.10g "
                                "n_trig=%d regime=r%03u\n",
                                (long long)d.bar_ts_ns, d.side, d.y_pred, d.threshold,
                                d.n_triggered, static_cast<unsigned>(d.winning_regime_code));
                    if (d.side != 0 && d.y_pred == 0.0) ++held;
                }
            }
        }
    }

    std::printf("\n[shadow] ticks=%zu bars=%zu warm_bars=%zu decisions_fired=%zu\n",
                ticks, bars, warm_bars, fired);
    if (bars == 0) { std::printf("=== FAIL: no bars were produced ===\n"); return 1; }
    if (!warm_announced) {
        // Run this with the PRODUCTION algo_params (warmup_fire_gate_bars=1080),
        // not a lowered gate — a replay under a small gate is exactly what let
        // the buffer-vs-counter defect through.
        std::printf("=== FAIL: never reached warmup (bars_fed=%d) — the gate never released ===\n",
                    core->barsBuffered());
        return 1;
    }
    if (warm_bars == 0) { std::printf("=== FAIL: no post-warmup bars scored ===\n"); return 1; }
    std::printf("=== PASS: live path ran end to end (tick -> bar -> warmup -> decide) ===\n");
    return 0;
}
