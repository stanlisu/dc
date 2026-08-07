// Unit test for the warmup fire gate — the C++ mirror of
// xmen/knull/tests/test_warmup_fire_gate.py.
//
// WHAT IT EXISTS TO CATCH
// -----------------------
// The port compared the gate against the BAR BUFFER's size. The buffer is
// trimmed to BUFFER_MAXLEN (1000) because the feature/regime panel is defined
// over exactly that window, so with the production gate of 1080 the comparison
// could never be satisfied: the shadow strategy on hydra ran 3d22h, logged
// "warming bars=1000" 103 times, and fired nothing. The reference counts bars
// CUMULATIVELY (`_count_bar_fed` / `_warmup_gate_ok`), which is why a gate above
// the buffer length is a legitimate config there.
//
// `warmup_over_buffer_maxlen` below is the case that reproduces the defect: it
// fails against the old `mBuf.size() >= gate` and passes against the counter.
//
// Standalone by construction — no TA-Lib, no LightGBM, no weights, no regime
// stack. That is the point: constructing a Core needs all four, which is why
// this logic had no cheap test before.
//
// Usage: warmup_gate_driver          (no arguments; exit 0 = pass)
#include "../src/warmup_gate.hpp"

#include <cstdio>
#include <deque>
#include <string>

using namespace mjolnir;

namespace {

// Mirrors core_impl.cpp's BUFFER_MAXLEN (it lives in that file's anonymous
// namespace, so it cannot be included). If the two ever drift the test still
// holds: the property under test is "a gate above the buffer length opens",
// not the specific number.
constexpr int BUFFER_MAXLEN = 1000;

int g_failures = 0;

void check(bool ok, const std::string& what)
{
    if (ok) {
        std::printf("  ok    %s\n", what.c_str());
    } else {
        std::printf("  FAIL  %s\n", what.c_str());
        ++g_failures;
    }
}

// ---------------------------------------------------------------------------

// Mirrors test_gate_blocks_until_enough_base_bars_then_passes.
void blocks_until_enough_bars()
{
    std::printf("[warmup_gate] blocks_until_enough_bars\n");
    WarmupGate g(3);
    check(!g.isWarm(), "cold before any bar");
    g.countBarFed();
    g.countBarFed();
    check(!g.isWarm(), "still cold at 2/3");
    g.countBarFed();
    check(g.isWarm(), "warm at 3/3");
    check(g.isWarm(), "stays warm once passed");
}

// Mirrors test_default_gate_value_boundary: exact, not off by one.
void boundary_is_exact()
{
    std::printf("[warmup_gate] boundary_is_exact\n");
    WarmupGate g(120);
    for (int i = 0; i < 119; ++i) g.countBarFed();
    check(!g.isWarm(), "cold at 119/120");
    check(g.barsFed() == 119, "barsFed reports 119");
    g.countBarFed();
    check(g.isWarm(), "warm at 120/120");
    check(g.barsFed() == 120, "barsFed reports 120");
}

// THE REGRESSION. A gate above BUFFER_MAXLEN must still open, and the count
// must keep climbing past the buffer length rather than saturating at it.
void warmup_over_buffer_maxlen()
{
    std::printf("[warmup_gate] warmup_over_buffer_maxlen (the hydra stall)\n");
    const int kGate = 1080;   // the production value, algo_params.hydra.json
    WarmupGate g(kGate);

    // Drive the gate alongside a buffer trimmed exactly as core_impl trims it,
    // so the assertions below are about the two DIVERGING — which is the bug.
    std::deque<int> buf;
    for (int i = 0; i < BUFFER_MAXLEN; ++i) {
        buf.push_back(i);
        g.countBarFed();
        while (static_cast<int>(buf.size()) > BUFFER_MAXLEN) buf.pop_front();
    }
    check(static_cast<int>(buf.size()) == BUFFER_MAXLEN, "buffer saturated at 1000");
    check(g.barsFed() == BUFFER_MAXLEN, "gate count also at 1000");
    check(!g.isWarm(), "not warm yet at 1000/1080");

    for (int i = 0; i < 80; ++i) {
        buf.push_back(i);
        g.countBarFed();
        while (static_cast<int>(buf.size()) > BUFFER_MAXLEN) buf.pop_front();
    }
    // The old code asked buf.size() >= 1080 here, which is false forever.
    check(static_cast<int>(buf.size()) == BUFFER_MAXLEN, "buffer STILL 1000 after trim");
    check(g.barsFed() == kGate, "gate count climbed past the buffer to 1080");
    check(g.isWarm(), "WARM at 1080 even though the buffer never exceeds 1000");

    for (int i = 0; i < 500; ++i) g.countBarFed();
    check(g.isWarm(), "stays warm");
    check(g.barsFed() == kGate + 500, "count keeps climbing — a stall stays visible");
}

// A zero gate is warm immediately; a negative one is a config error, not a
// silently clamped zero.
void degenerate_gates()
{
    std::printf("[warmup_gate] degenerate_gates\n");
    WarmupGate zero(0);
    check(zero.isWarm(), "gate 0 is warm with no bars");

    bool raised = false;
    try {
        WarmupGate bad(-1);
        (void)bad;
    } catch (const std::exception&) {
        raised = true;
    }
    check(raised, "negative gate raises rather than clamping");
}

} // namespace

int main()
{
    blocks_until_enough_bars();
    boundary_is_exact();
    warmup_over_buffer_maxlen();
    degenerate_gates();

    if (g_failures != 0) {
        std::printf("\n=== FAIL: %d warmup-gate assertion(s) ===\n", g_failures);
        return 1;
    }
    std::printf("\n=== PASS: warmup gate counts bars fed, not buffer occupancy ===\n");
    return 0;
}
