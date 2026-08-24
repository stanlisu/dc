// Order-path arithmetic gate for AgamottoStrategy.
//
// WHY THIS EXISTS AS A SEPARATE DRIVER. The order path lives on the SENTINEL
// side, in AgamottoStrategy, which derives from the vendor's AlgoBase. AlgoBase
// has out-of-line ctor/dtor/send_order and ltp_strat_sdk/lib/ is gitignored, so
// ANY member function of the strategy is unlinkable in a test — the whole class
// is untestable in-process. The arithmetic that decides whether money moves is
// therefore kept in order_path_math.hpp, which includes nothing but <cmath> and
// <cstdint>, and this driver links it alone.
//
// It deliberately does NOT link agamotto_core: no ta-lib, no weights, no
// AGAMOTTO_CORE_GITSHA. The property under test is the sizing/pricing/guard
// arithmetic, not the model.
//
// Everything the strategy still owns after this — the root_.get<T> config
// reads, send_order() itself, and the OMS_TRADE fill accounting — cannot be
// reached from here. Its only rehearsal is order_path_dry_run against a live
// feed, which is why that mode exists.
#include <cstdio>
#include <cmath>
#include <string>

#include "order_path_math.hpp"

using namespace agamotto::orderpath;

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

// A book that is fresh, two-sided and uncrossed, so each test perturbs exactly
// one thing rather than starting from something already invalid.
Book goodBook()
{
    Book b{};
    b.bid = 99.0;
    b.ask = 101.0;
    b.recv_ns = 99ull * 1000ull * 1000ull * 1000ull;   // 99s: 1s old at NOW_NS
    return b;
}

OrderConfig btcCfg()
{
    OrderConfig c{};
    c.capital_usd = 110.0;
    c.max_position_notional_usd = 110.0;
    c.max_rungs_per_ladder = 5;
    c.price_tick = 0.10;
    c.qty_step = 0.001;
    c.cross_pct = 0.0005;
    c.quote_stale_ns = 5ull * 1000ull * 1000ull * 1000ull;   // 5s
    c.reverse = 1;
    return c;
}

const uint64_t NOW_NS = 100ull * 1000ull * 1000ull * 1000ull;   // 100s

}  // namespace

int main()
{
    std::printf("=== order path arithmetic ===\n");

    // ---- [1] floorToStep FLOORS, never rounds up -------------------------
    // Rounding up would push notional past the cap that was just checked, so an
    // over-cap order is created by the very function meant to bound it.
    std::printf("[1] floorToStep\n");
    checkClose(floorToStep(0.0019, 0.001), 0.001, 1e-12, "0.0019 -> 0.001 (floor)");
    checkClose(floorToStep(0.0010, 0.001), 0.001, 1e-12, "exact step is preserved");
    checkClose(floorToStep(0.0009, 0.001), 0.0,   1e-12, "below one step -> 0");
    // 0.29/0.01 is 28.999999999999996 in binary; a naive floor gives 0.28.
    checkClose(floorToStep(0.29, 0.01), 0.29, 1e-12, "FP boundary 0.29/0.01 stays 0.29");
    checkClose(floorToStep(1.0, 0.0), 0.0, 1e-12, "non-positive step -> 0, never divide");

    // ---- [2] net_count is capped BEFORE qty derivation --------------------
    // agamotto_bridge.py:643-646 caps first and says why: qty is
    // CAPITAL*net_count, so clamping afterwards leaves the notional uncapped.
    std::printf("[2] cappedNetCount\n");
    check(cappedNetCount(3, 5) == 3,   "3 votes under the cap is untouched");
    check(cappedNetCount(6, 5) == 5,   "6 votes capped to 5");
    check(cappedNetCount(-6, 5) == -5, "cap applies to magnitude, sign kept");
    check(cappedNetCount(0, 5) == 0,   "no net vote stays 0");

    // ---- [3] reverse flips the side; anything but +/-1 is refused ---------
    // trading.py multiplies a QUANTITY by reverse and never validates it, so 0
    // is a permanently flat bot and 2 a silent doubling of live size.
    std::printf("[3] reverse\n");
    {
        OrderConfig c = btcCfg();
        c.reverse = -1;
        const Plan p = planOrder(1, 0, 0.0, goodBook(), c, NOW_NS);
        check(p.action == Action::SEND, "reverse=-1 still trades");
        check(p.side == -1, "reverse=-1 turns a long vote into a SELL");
    }
    {
        OrderConfig c = btcCfg();
        c.reverse = 0;
        const Plan p = planOrder(1, 0, 0.0, goodBook(), c, NOW_NS);
        check(p.action == Action::HALT, "reverse=0 refuses rather than sitting flat");
    }
    {
        OrderConfig c = btcCfg();
        c.reverse = 2;
        const Plan p = planOrder(1, 0, 0.0, goodBook(), c, NOW_NS);
        check(p.action == Action::HALT, "reverse=2 refuses rather than doubling size");
    }

    // ---- [4] sub-lot target produces no order -----------------------------
    std::printf("[4] sub-lot\n");
    {
        OrderConfig c = btcCfg();
        c.capital_usd = 0.05;            // 0.05/100 = 0.0005 < step 0.001
        const Plan p = planOrder(1, 0, 0.0, goodBook(), c, NOW_NS);
        check(p.action == Action::NONE, "target below one lot step -> no order");
    }

    // ---- [5] the book must be fresh, two-sided and uncrossed --------------
    // Refuse rather than reuse: a stale book is not a price we can trust, and
    // reusing one prices a real order off a market that has moved.
    std::printf("[5] book guards\n");
    {
        Book b = goodBook();
        b.recv_ns = 1;                              // ~100s old vs a 5s tolerance
        check(planOrder(1, 0, 0.0, b, btcCfg(), NOW_NS).action == Action::NONE,
              "stale book -> no order");
    }
    {
        Book b = goodBook();
        b.bid = 0.0;
        check(planOrder(1, 0, 0.0, b, btcCfg(), NOW_NS).action == Action::NONE,
              "one-sided book (no bid) -> no order");
    }
    {
        Book b = goodBook();
        b.ask = 0.0;
        check(planOrder(1, 0, 0.0, b, btcCfg(), NOW_NS).action == Action::NONE,
              "one-sided book (no ask) -> no order");
    }
    {
        Book b = goodBook();
        b.bid = 102.0;                              // crossed
        check(planOrder(1, 0, 0.0, b, btcCfg(), NOW_NS).action == Action::NONE,
              "crossed book -> no order");
    }
    {
        // Clock skew: a book stamped in the future must not read as infinitely
        // stale via unsigned wraparound.
        Book b = goodBook();
        b.recv_ns = NOW_NS + 1000;
        check(planOrder(1, 0, 0.0, b, btcCfg(), NOW_NS).action == Action::SEND,
              "book stamped slightly ahead is fresh, not wrapped-around ancient");
    }

    // ---- [6] reduce_only IFF the target is flat ---------------------------
    // A flip must NOT be reduce_only or the exchange clips it at zero and
    // leaves us flat instead of reversed.
    std::printf("[6] reduce_only\n");
    {
        const Plan p = planOrder(1, 0, 0.0, goodBook(), btcCfg(), NOW_NS);
        check(p.action == Action::SEND && !p.reduce_only, "opening is not reduce_only");
    }
    {
        // long 1.1 -> target flat
        const Plan p = planOrder(0, 0, 1.1, goodBook(), btcCfg(), NOW_NS);
        check(p.action == Action::SEND, "flat target from a long position trades");
        check(p.side == -1, "closing a long SELLS");
        check(p.reduce_only, "pure close IS reduce_only");
    }
    {
        // long 1.1 -> target short: a FLIP
        const Plan p = planOrder(0, 1, 1.1, goodBook(), btcCfg(), NOW_NS);
        check(p.action == Action::SEND, "flip trades");
        check(p.side == -1, "long -> short SELLS");
        check(!p.reduce_only, "a FLIP is NOT reduce_only");
    }

    // ---- [7] cap breach HALTS, never clamps -------------------------------
    // Clamping a derived quantity converts the bug into a plausible in-range
    // number, so the bug never surfaces (CLAUDE.md).
    std::printf("[7] notional cap\n");
    {
        OrderConfig c = btcCfg();
        c.capital_usd = 110.0;
        c.max_position_notional_usd = 110.0;
        // 3 net votes at $110 each = $330 of intent against a $110 cap.
        const Plan p = planOrder(3, 0, 0.0, goodBook(), c, NOW_NS);
        check(p.action == Action::HALT, "over-cap intent HALTS");
        check(p.qty == 0.0, "a halted plan carries no quantity to send");
    }
    {
        OrderConfig c = btcCfg();
        c.max_position_notional_usd = 330.0;
        const Plan p = planOrder(3, 0, 0.0, goodBook(), c, NOW_NS);
        check(p.action == Action::SEND, "the same order under a larger cap sends");
    }

    // ---- [8] IOC price crosses the right touch, snapped to the tick -------
    std::printf("[8] ioc price\n");
    {
        const Plan p = planOrder(1, 0, 0.0, goodBook(), btcCfg(), NOW_NS);
        // buy crosses the ASK upward: 101 * 1.0005 = 101.0505 -> tick 0.10
        checkClose(p.px, 101.10, 1e-9, "BUY prices through the ask, snapped up-grid");
        check(p.px > goodBook().ask, "a crossing BUY is above the ask");
    }
    {
        const Plan p = planOrder(0, 1, 0.0, goodBook(), btcCfg(), NOW_NS);
        // sell crosses the BID downward: 99 * 0.9995 = 98.9505 -> tick 0.10
        checkClose(p.px, 98.90, 1e-9, "SELL prices through the bid, snapped down-grid");
        check(p.px < goodBook().bid, "a crossing SELL is below the bid");
    }
    {
        // The OMS rejects any price off the tick grid (LTP 401015), so the snap
        // is not cosmetic.
        const Plan p = planOrder(1, 0, 0.0, goodBook(), btcCfg(), NOW_NS);
        const double ticks = p.px / btcCfg().price_tick;
        checkClose(ticks, std::round(ticks), 1e-6, "price lands exactly on the tick grid");
    }

    // ---- [9] already at target -> no churn --------------------------------
    std::printf("[9] no churn\n");
    {
        // target for 1 net vote at mid 100 = 110/100 = 1.1
        const Plan p = planOrder(1, 0, 1.1, goodBook(), btcCfg(), NOW_NS);
        check(p.action == Action::NONE, "already at target -> no order");
    }
    {
        const Plan p = planOrder(1, 0, 1.1005, goodBook(), btcCfg(), NOW_NS);
        check(p.action == Action::NONE, "within one lot of target -> no order");
    }
    {
        const Plan p = planOrder(0, 0, 0.0, goodBook(), btcCfg(), NOW_NS);
        check(p.action == Action::NONE, "flat and no vote -> no order");
    }

    // ---- [10] sizing matches the reference --------------------------------
    // qty = floor_step(CAPITAL * net_count / price, step), capped first.
    std::printf("[10] sizing vs reference\n");
    {
        OrderConfig c = btcCfg();
        c.max_position_notional_usd = 1e9;   // isolate sizing from the cap
        const Plan p = planOrder(2, 0, 0.0, goodBook(), c, NOW_NS);
        // 110*2/100 = 2.2
        checkClose(p.qty, 2.2, 1e-9, "2 net votes -> 2x CAPITAL/mid");
    }
    {
        OrderConfig c = btcCfg();
        c.max_position_notional_usd = 1e9;
        const Plan p = planOrder(9, 0, 0.0, goodBook(), c, NOW_NS);
        // capped to 5: 110*5/100 = 5.5
        checkClose(p.qty, 5.5, 1e-9, "9 votes capped to 5 BEFORE sizing");
    }

    std::printf("\n=== %s: %d checks, %d failures ===\n",
                g_failures == 0 ? "ORDER PATH PASS" : "ORDER PATH FAIL",
                g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
