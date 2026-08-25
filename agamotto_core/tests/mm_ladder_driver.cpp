// MarketMaker ladder reducer — the gate for sentinel's MM execution port.
//
// WHY A PURE REDUCER, TESTED HERE. AgamottoStrategy cannot be linked in a test
// binary at all: AlgoBase has an out-of-line ctor and send_order, and
// ltp_strat_sdk/lib/ is gitignored. Anything left as a member function of the
// strategy has no test but a live venue -- unacceptable for MM, where the
// decision was taken to build the FULL lifecycle before any live order.
//
// So mm_ladder.hpp is <cmath>/<cstdint>/<array> only, and every rule below is
// one that knull learned from a real incident. Each has a mutant in
// run_mm_ladder_mutants.sh; a rule without a mutant is a rule nobody has
// checked.
#include <cstdio>
#include <cmath>
#include <string>

#include "mm_ladder.hpp"

using namespace agamotto::mm;

namespace {

int g_failures = 0;
int g_checks = 0;

void check(bool ok, const std::string& what)
{
    ++g_checks;
    if (!ok) {
        ++g_failures;
        std::printf("  FAIL  %s\n", what.c_str());
    }
}

void checkClose(double got, double want, double tol, const std::string& what)
{
    ++g_checks;
    if (!(std::fabs(got - want) <= tol)) {
        ++g_failures;
        std::printf("  FAIL  %s: got %.10f, want %.10f (tol %g)\n",
                    what.c_str(), got, want, tol);
    }
}

Book book(double bid, double ask, double last = 0.0)
{
    Book b{};
    b.bid = bid;
    b.ask = ask;
    b.last = last;
    b.recv_ns = 100ull * 1000000000ull;
    return b;
}

// Mirrors pred_agamotto.base.15m_1/setting.json.
Config cfg()
{
    Config c{};
    c.capital_usd = 110.0;
    c.max_rungs_per_ladder = 5;
    c.ladder = 2;
    c.ladder_bps = 1.0;
    c.aim_bps_long = 50.0;
    c.aim_bps_short = 50.0;
    c.passive_sec = 60.0;
    c.crossing_sec = 840.0;
    c.exit_depth_max = 3;
    c.exit_rungs = 5;
    c.reprice_open_sec = 10.0;
    c.reprice_close_sec = 10.0;
    c.entry_timeout_sec = 900.0;
    c.stoploss_frac = 0.005;
    c.price_tick = 0.10;
    c.qty_step = 0.001;
    c.min_notional = 50.0;
    c.quote_stale_ns = 5ull * 1000000000ull;
    return c;
}

const uint64_t NOW = 100ull * 1000000000ull;

}  // namespace

int main()
{
    std::printf("=== mm ladder reducer ===\n");

    // ---- [1] ENTRY ANCHOR AND THE SIGNAL CLAMP -----------------------------
    // BUY  anchor = min(bid, signal_close)
    // SELL anchor = max(ask, signal_close)
    // "Never quote outside the signal price." symbiote/common.py:135-142.
    std::printf("[1] entry anchor + signal clamp\n");
    checkClose(entryAnchor(+1, book(100.0, 100.1), 100.5), 100.0, 1e-9,
               "BUY takes the bid when the bid is inside the signal");
    checkClose(entryAnchor(+1, book(100.0, 100.1), 99.5), 99.5, 1e-9,
               "BUY clamped DOWN to the signal close");
    checkClose(entryAnchor(-1, book(100.0, 100.1), 99.5), 100.1, 1e-9,
               "SELL takes the ask when the ask is inside the signal");
    checkClose(entryAnchor(-1, book(100.0, 100.1), 100.5), 100.5, 1e-9,
               "SELL clamped UP to the signal close");
    checkClose(entryAnchor(+1, book(100.0, 100.1), 0.0), 100.0, 1e-9,
               "no signal price -> plain touch, no clamp");

    // ---- [2] RUNGS WALK AWAY FROM THE MARKET, AFTER THE CLAMP -------------
    // rung i = anchor -/+ i*tick, deeper into our own side. common.py:144-153.
    std::printf("[2] rung walk\n");
    {
        double px[8] = {0};
        const int n = rungPrices(+1, 100.0, 3, cfg().price_tick, px, 8);
        check(n == 3, "BUY produced 3 rungs");
        checkClose(px[0], 100.0, 1e-9, "BUY rung 0 at the anchor");
        checkClose(px[1], 99.9, 1e-9, "BUY rung 1 one tick BELOW");
        checkClose(px[2], 99.8, 1e-9, "BUY rung 2 two ticks below");
    }
    {
        double px[8] = {0};
        rungPrices(-1, 100.1, 3, cfg().price_tick, px, 8);
        checkClose(px[1], 100.2, 1e-9, "SELL rung 1 one tick ABOVE");
    }

    // ---- [3] MM AIM ANCHOR: DIRECTION-AWARE ROUNDING ----------------------
    // LONG  SELL at max(ceil(ask), ceil(cost*(1+aim)))
    // SHORT BUY  at min(floor(bid), floor(cost*(1-aim)))
    // Round-to-nearest turns the profit aim from a FLOOR into a stochastic
    // average -- market_maker.py:249-254, TON closed at 0.89bp vs a 1bp target.
    std::printf("[3] mm aim anchor\n");
    {
        // cost 100, aim 50bps -> 100.5 exactly; ask below it, so aim wins.
        const double a = aimAnchor(+1, 100.0, book(100.0, 100.1), cfg());
        checkClose(a, 100.5, 1e-9, "LONG aim 50bps above cost");
        check(a >= 100.5 - 1e-12, "LONG aim never rounds BELOW the target");
    }
    {
        // cost 100.03 -> 100.53015; ceil to 0.10 grid = 100.60, never 100.50.
        const double a = aimAnchor(+1, 100.03, book(100.0, 100.1), cfg());
        checkClose(a, 100.6, 1e-9, "LONG aim CEILs to the tick grid");
    }
    {
        const double a = aimAnchor(-1, 100.0, book(100.0, 100.1), cfg());
        checkClose(a, 99.5, 1e-9, "SHORT aim 50bps below cost");
    }
    {
        // cost 99.97 -> 99.47015; floor to grid = 99.40, never 99.50.
        const double a = aimAnchor(-1, 99.97, book(100.0, 100.1), cfg());
        checkClose(a, 99.4, 1e-9, "SHORT aim FLOORs to the tick grid");
    }
    {
        // Ask above the aim: post-only would cross at the aim, so the ask wins.
        const double a = aimAnchor(+1, 100.0, book(101.0, 101.1), cfg());
        checkClose(a, 101.1, 1e-9, "LONG aim never rests below the ask");
    }

    // ---- [4] EXIT ANCHOR IS THE CONSERVATIVE SIDE -------------------------
    // SELL min(ask, last); BUY max(bid, last). base_executor.py:1859-1878.
    std::printf("[4] exit anchor\n");
    checkClose(exitAnchor(+1, book(100.0, 100.1, 100.05)), 100.05, 1e-9,
               "long exit SELLs at min(ask, last)");
    checkClose(exitAnchor(-1, book(100.0, 100.1, 100.05)), 100.05, 1e-9,
               "short exit BUYs at max(bid, last)");
    checkClose(exitAnchor(+1, book(100.0, 100.1, 0.0)), 100.1, 1e-9,
               "no last print -> degrade to the touch");

    // ---- [5] EXIT PHASE B DEPTH WALK --------------------------------------
    // depth = 1 + elapsed_in_B / (CROSSING_SEC/3.0), capped at EXIT_DEPTH_MAX.
    // The /3.0 is HARDCODED, not derived from the cap. execution_style.py:300.
    std::printf("[5] phase B depth walk\n");
    checkClose(static_cast<double>(crossingDepth(0.0, cfg())), 1.0, 0,
               "depth 1 at the start of phase B");
    checkClose(static_cast<double>(crossingDepth(279.0, cfg())), 1.0, 0,
               "still depth 1 at 279s (step is 280s)");
    checkClose(static_cast<double>(crossingDepth(280.0, cfg())), 2.0, 0,
               "depth 2 at 280s");
    checkClose(static_cast<double>(crossingDepth(560.0, cfg())), 3.0, 0,
               "depth 3 at 560s");
    checkClose(static_cast<double>(crossingDepth(5000.0, cfg())), 3.0, 0,
               "capped at EXIT_DEPTH_MAX, never deeper");

    // ---- [6] THE KEEP-IF-MATCH DIFF ---------------------------------------
    // Idempotent: an unchanged ladder emits ZERO ops and keeps queue position.
    // A one-tick shift touches exactly the offending rung. Cancel-storms on
    // sumo (~1.2k OKX 51400/day) came from rebuilding instead of matching.
    std::printf("[6] keep-if-match diff\n");
    {
        LadderState st{};
        st.n_live = 3;
        for (int i = 0; i < 3; ++i) {
            st.live[i].uid = 100 + i;
            st.live[i].price = 100.0 - i * 0.1;
            st.live[i].qty = 0.001;
            st.live[i].active = true;
        }
        double wp[8] = {100.0, 99.9, 99.8};
        double wq[8] = {0.001, 0.001, 0.001};
        Intents in = diffLadder(st, wp, wq, 3, cfg());
        check(in.n_cancel == 0 && in.n_place == 0,
              "identical ladder -> ZERO ops (queue position preserved)");

        double wp2[8] = {100.1, 99.9, 99.8};
        Intents in2 = diffLadder(st, wp2, wq, 3, cfg());
        check(in2.n_cancel == 1 && in2.n_place == 1,
              "one-tick shift -> exactly 1 cancel + 1 place");
        check(in2.cancel_uid[0] == 100, "cancels the rung that moved, not all");
    }

    // ---- [7] POSITION CAP -------------------------------------------------
    // filled + live_FULL_qty + new > qty*LADDER*(1+eps). A partially filled
    // rung is counted TWICE, deliberately. ladder_math.py:104-138.
    std::printf("[7] position cap\n");
    {
        LadderState st{};
        st.tier_capacity = 0.005;
        st.filled_qty = 0.005;
        st.n_live = 0;
        check(!wouldExceedCap(st, 0.005, cfg()),
              "second tier fits under LADDER=2");
        check(wouldExceedCap(st, 0.006, cfg()),
              "beyond LADDER*capacity is refused");
    }
    {
        LadderState st{};
        st.tier_capacity = 0.005;
        st.filled_qty = 0.003;
        st.n_live = 1;
        st.live[0].qty = 0.005;      // FULL ordered qty counts
        st.live[0].filled = 0.003;   // even though 0.003 is already in filled
        st.live[0].active = true;
        check(wouldExceedCap(st, 0.004, cfg()),
              "partially-filled rung counts twice -- conservative by design");
    }
    {
        LadderState st{};
        st.tier_capacity = 0.0;
        check(!wouldExceedCap(st, 999.0, cfg()),
              "capacity 0 lets the FIRST ladder build through");
    }

    // ---- [8] TIER GATE IS CONJUNCTIVE AND STICKY --------------------------
    // Fires only when the last tier is FULLY filled AND the market moved
    // LADDER_BPS against cost. Once armed the trigger never clears until FLAT.
    std::printf("[8] tier gate\n");
    {
        LadderState st{};
        st.side = +1;
        st.tier_capacity = 0.005;
        st.filled_qty = 0.004;          // tier NOT full
        st.avg_cost = 100.0;
        st.level = 1;
        check(!tierShouldFire(st, book(99.0, 99.1), cfg()),
              "partial tier does not fire even when the market moved");
        st.filled_qty = 0.005;          // tier full
        check(!tierShouldFire(st, book(100.0, 100.1), cfg()),
              "full tier alone does not fire without the move");
        check(tierShouldFire(st, book(99.0, 99.1), cfg()),
              "full tier AND market moved -> fires");
    }
    {
        LadderState st{};
        st.side = +1;
        st.tier_capacity = 0.005;
        st.filled_qty = 0.005;
        st.avg_cost = 100.0;
        st.level = 1;
        st.tier_triggered[1] = true;    // armed earlier
        check(tierShouldFire(st, book(100.0, 100.1), cfg()),
              "trigger is STICKY: still fires after the market recovers");
    }
    {
        LadderState st{};
        st.side = +1;
        st.tier_capacity = 0.005;
        st.filled_qty = 0.010;
        st.avg_cost = 100.0;
        st.level = 2;                   // already at LADDER
        check(!tierShouldFire(st, book(99.0, 99.1), cfg()),
              "never fires past LADDER tiers");
    }

    // ---- [9] FILL ACCOUNTING ----------------------------------------------
    // The part that permanently corrupts position when wrong, and the reason
    // this is a reducer at all.
    std::printf("[9] fill accounting\n");
    {
        LadderState st{};
        st.side = +1;
        st.n_live = 1;
        st.live[0].uid = 42;
        st.live[0].qty = 0.002;
        st.live[0].active = true;

        Response r{};
        r.uid = 42;
        r.kind = Response::TRADE;
        r.qty = 0.001;
        r.price = 100.0;
        LadderState s1 = applyResponse(st, r);
        checkClose(s1.filled_qty, 0.001, 1e-12, "partial fill accrues");
        checkClose(s1.avg_cost, 100.0, 1e-9, "avg cost from the first fill");
        check(s1.live[0].active, "rung still live after a partial");

        r.qty = 0.001;
        r.price = 101.0;
        LadderState s2 = applyResponse(s1, r);
        checkClose(s2.filled_qty, 0.002, 1e-12, "second partial completes it");
        checkClose(s2.avg_cost, 100.5, 1e-9, "avg cost is qty-weighted");
        check(!s2.live[0].active, "fully filled rung leaves the live set");
    }
    {
        // A fill for a uid we do not hold must never move position.
        LadderState st{};
        st.n_live = 0;
        Response r{};
        r.uid = 999;
        r.kind = Response::TRADE;
        r.qty = 5.0;
        LadderState s = applyResponse(st, r);
        checkClose(s.filled_qty, 0.0, 1e-12,
                   "fill for an unknown uid is ignored, not booked");
    }
    {
        // LATE trade on a cancelled uid. The venue can fill between our cancel
        // and its acceptance; the qty is REAL and must be booked.
        LadderState st{};
        st.side = +1;
        st.n_live = 1;
        st.live[0].uid = 7;
        st.live[0].qty = 0.002;
        st.live[0].active = true;
        st.live[0].cancelling = true;
        Response r{};
        r.uid = 7;
        r.kind = Response::TRADE;
        r.qty = 0.001;
        r.price = 100.0;
        LadderState s = applyResponse(st, r);
        checkClose(s.filled_qty, 0.001, 1e-12,
                   "late fill on a CANCELLING rung is still booked");
    }
    {
        // An implausible fill must halt rather than corrupt the position.
        LadderState st{};
        st.n_live = 1;
        st.live[0].uid = 3;
        st.live[0].qty = 0.001;
        st.live[0].active = true;
        Response r{};
        r.uid = 3;
        r.kind = Response::TRADE;
        r.qty = 99.0;
        LadderState s = applyResponse(st, r);
        check(s.halted, "fill larger than the order HALTS");
        checkClose(s.filled_qty, 0.0, 1e-12, "and books nothing");
    }

    // ---- [10] REJECTS AND THROTTLE ----------------------------------------
    std::printf("[10] rejects\n");
    {
        LadderState st{};
        st.n_live = 1;
        st.live[0].uid = 5;
        st.live[0].active = true;
        Response r{};
        r.uid = 5;
        r.kind = Response::REJECT;
        r.error_code = -9996;           // THROTTLE_LIMIT
        LadderState s = applyResponse(st, r);
        check(!s.live[0].active, "a rejected rung leaves the live set");
        check(!s.halted, "one throttle reject does not halt the ladder");
        check(s.rejects == 1, "rejects are counted");
    }

    // ---- [11] THE STATE MACHINE ------------------------------------------
    // FLAT -> ENTERING -> OPEN -> EXITING -> FLAT. Every transition below is
    // one knull has a named event for; getting the entry timeout fork wrong
    // strands a partially-filled position in ENTERING forever.
    std::printf("[11] state machine\n");
    {
        LadderState st{};
        check(st.phase == Phase::FLAT, "starts FLAT");
        // Placing the first entry rung is what moves us, not the signal.
        LadderState s1 = onPlaced(st, /*uid*/1, 100.0, 0.001, +1, 1000.0);
        check(s1.phase == Phase::ENTERING, "first placed rung -> ENTERING");
        check(s1.side == +1, "side is latched on the first placement");
        checkClose(s1.entered_at, 1000.0, 1e-9, "entry clock starts");
        checkClose(s1.tier_capacity, 0.001, 1e-12,
                   "tier capacity latches to the first ladder's size");

        // A fill during ENTERING opens the position.
        Response r{};
        r.uid = 1; r.kind = Response::TRADE; r.qty = 0.001; r.price = 100.0;
        LadderState s2 = applyResponse(s1, r);
        check(s2.phase == Phase::OPEN, "first fill -> OPEN");
    }
    {
        // ENTRY TIMEOUT with nothing filled -> back to FLAT.
        LadderState st{};
        st.phase = Phase::ENTERING;
        st.entered_at = 1000.0;
        st.filled_qty = 0.0;
        LadderState s = applyClock(st, 1000.0 + 900.0, cfg());
        check(s.phase == Phase::FLAT, "entry timeout, nothing filled -> FLAT");
    }
    {
        // ENTRY TIMEOUT with a partial fill -> OPEN, not FLAT. Going FLAT here
        // abandons a real position the venue is holding.
        LadderState st{};
        st.phase = Phase::ENTERING;
        st.entered_at = 1000.0;
        st.filled_qty = 0.0005;
        st.side = +1;
        LadderState s = applyClock(st, 1000.0 + 900.0, cfg());
        check(s.phase == Phase::OPEN, "entry timeout WITH a fill -> OPEN");
    }
    {
        LadderState st{};
        st.phase = Phase::ENTERING;
        st.entered_at = 1000.0;
        LadderState s = applyClock(st, 1000.0 + 899.0, cfg());
        check(s.phase == Phase::ENTERING, "not yet timed out at 899s");
    }
    {
        // OPEN -> EXITING happens on the exit ladder being placed.
        LadderState st{};
        st.phase = Phase::OPEN;
        st.side = +1;
        st.filled_qty = 0.002;
        LadderState s = onExitStarted(st, 2000.0);
        check(s.phase == Phase::EXITING, "exit placed -> EXITING");
        checkClose(s.exit_started_at, 2000.0, 1e-9, "exit clock starts");
    }
    {
        // Reduce to net zero -> FLAT, and the state must be WIPED, not
        // patched. knull replaces the position object at three sites for this
        // reason; leaving stale fields caused the 2026-06-18 phantom stoploss.
        LadderState st{};
        st.phase = Phase::EXITING;
        st.side = +1;
        st.filled_qty = 0.001;
        st.avg_cost = 100.0;
        st.level = 2;
        st.tier_triggered[1] = true;
        st.n_live = 1;
        st.live[0].uid = 9; st.live[0].qty = 0.001; st.live[0].active = true;
        Response r{};
        r.uid = 9; r.kind = Response::TRADE; r.qty = 0.001; r.price = 101.0;
        r.reduce_only = true;
        LadderState s = applyResponse(st, r);
        check(s.phase == Phase::FLAT, "net zero -> FLAT");
        checkClose(s.filled_qty, 0.0, 1e-12, "position wiped");
        checkClose(s.avg_cost, 0.0, 1e-12, "avg cost wiped");
        check(s.level == 1, "tier level reset");
        check(!s.tier_triggered[1], "sticky tier triggers cleared on FLAT");
        check(s.side == 0, "side cleared");
    }

    // ---- [11b] COST BASIS AND TIER CAPACITY ------------------------------
    // Two rules with no assertion until a mutant survived. Both silently
    // corrupt sizing rather than failing loudly.
    std::printf("[11b] cost basis + tier capacity\n");
    {
        // An EXIT fill must not move the cost basis. The aim ladder is priced
        // off avg_cost, so folding exit prices in makes the ladder chase its
        // own fills -- it would walk away as it filled.
        LadderState st{};
        st.phase = Phase::OPEN;
        st.side = +1;
        st.filled_qty = 0.002;
        st.avg_cost = 100.0;
        st.n_live = 1;
        st.live[0].uid = 11; st.live[0].qty = 0.001; st.live[0].active = true;
        Response r{};
        r.uid = 11; r.kind = Response::TRADE; r.qty = 0.001; r.price = 105.0;
        r.reduce_only = true;
        LadderState s = applyResponse(st, r);
        checkClose(s.avg_cost, 100.0, 1e-9,
                   "a reduce_only fill leaves the cost basis alone");
        checkClose(s.filled_qty, 0.001, 1e-12, "and reduces the position");
    }
    {
        // tier_capacity latches to the FIRST ladder only. If it kept growing,
        // every size-up would raise the cap it is measured against and the
        // position cap would never bind.
        LadderState st{};
        LadderState s1 = onPlaced(st, 1, 100.0, 0.001, +1, 1000.0);
        LadderState s2 = onPlaced(s1, 2, 99.9, 0.001, +1, 1000.0);
        checkClose(s2.tier_capacity, 0.002, 1e-12,
                   "capacity accrues across the FIRST ladder's rungs");
        s2.level = 2;                     // a size-up tier has fired
        LadderState s3 = onPlaced(s2, 3, 99.0, 0.005, +1, 1100.0);
        checkClose(s3.tier_capacity, 0.002, 1e-12,
                   "capacity does NOT grow once past level 1");
    }

    // ---- [12] EXIT PHASE FORK --------------------------------------------
    // Phase A rests passively for PASSIVE_SEC; Phase B walks depth. The fork
    // is on elapsed-since-exit, not on a stored flag.
    std::printf("[12] exit phase fork\n");
    {
        LadderState st{};
        st.phase = Phase::EXITING;
        st.exit_started_at = 1000.0;
        check(!inCrossingPhase(st, 1000.0 + 59.0, cfg()),
              "still passive at 59s");
        check(inCrossingPhase(st, 1000.0 + 60.0, cfg()),
              "crossing from PASSIVE_SEC");
        checkClose(phaseBElapsed(st, 1000.0 + 340.0, cfg()), 280.0, 1e-9,
                   "phase B elapsed excludes the passive window");
    }

    std::printf("\n=== %s: %d checks, %d failures ===\n",
                g_failures == 0 ? "MM LADDER PASS" : "MM LADDER FAIL",
                g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
