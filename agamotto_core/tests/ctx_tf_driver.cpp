// ABI 6 GATE — context timeframes (orb), end to end through createCore().
//
// orb regimes name each atom's timeframe (`1d_r012_and_1h_r028_long`). An atom
// on a CONTEXT timeframe is evaluated on that timeframe's own panel, engineered
// from native klines handed in through ingestContextBars, at the row orb's
// research takes with a backward as-of: the newest context bar whose close
// (open + tf) is at or before the base bar's OPEN. It must be the bar closing at
// floor(T / tf) * tf, or the regime does not fire (stale, counted).
//
// The expectation is computed INDEPENDENTLY of the core's context path: this
// driver re-engineers the same context bars with engineerFeaturesContext, picks
// the row by its OWN as-of, and reads atomMask there. A core that used the
// wrong row, the wrong timeframe's panel, or the newest bar instead of the
// as-of one disagrees on some atom, and the sensitivity checks below PROVE the
// sweep would see each of those: a fixture on which the shifted row or the
// wrong panel happened to agree everywhere would pass a broken core.
//
// Asserted:
//   [0] createCore refuses a context timeframe equal to bar_sec, off the
//       minute grid, repeated, or more than 8 of them; accepts 0 (agamotto).
//   [1] setRegimeStack refuses an atom on an unconfigured timeframe, a regime
//       mixing prefixed and bare atoms, and a timeframe in an unused slot;
//       loads weights by the PREFIXED directory name.
//   [2] ingestContextBars refuses an unknown timeframe, an off-grid bar and a
//       non-increasing series (changing nothing); drops a still-open bar by
//       time; keeps at most PANEL_BARS; accepts a short (whole-history) 1d.
//   [3] fresh lookups: every single-atom 1h and 1d regime, both positions,
//       fires exactly when atomMask holds at the as-of row.
//   [4] THE ONE-BAR LAG: with the 1h bar that closes together with the base
//       bar already ingested, the base bar still reads the PREVIOUS 1h bar.
//   [5] stale: a base bar whose needed 1h bar is missing fires NO 1h regime and
//       counts one stale lookup; the 1d regimes are unaffected.
//   [6] mixed 3-timeframe conjunctions = AND of their parts, where the base
//       part is read off a base-only regime in the same stack.
//
//   ./ctx_tf_driver        # exits nonzero on any failure
#include "agamotto_core.hpp"
#include "feature_engine.hpp"
#include "model_runner.hpp"
#include "regime_gate.hpp"

#include <sys/stat.h>

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace agamotto;

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

constexpr int     kBarSec = 900;
constexpr int     kWarmup = static_cast<int>(PANEL_BARS) + 1;
constexpr int64_t kP = kBarSec * 1000LL;
constexpr int64_t kH = 3600 * 1000LL;
constexpr int64_t kD = 86400 * 1000LL;
constexpr int64_t kProduct = 2000000170LL;
constexpr int64_t kBase = 1767225600000LL;   // 2026-01-01T00:00:00Z, on every grid

struct Walk {
    uint64_t s;
    explicit Walk(uint64_t seed) : s(seed) {}
    double next()
    {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        return static_cast<double>(s >> 11) / static_cast<double>(1ULL << 53);
    }
};

KlineBar makeBar(int64_t open_ms, int64_t period_ms, double& px, Walk& w)
{
    // Larger moves than a 15m walk, so the context panel's predicates take
    // both values over a few hundred rows.
    px *= 1.0 + (w.next() - 0.5) * 0.02;
    KlineBar b{};
    b.bucket_open_ms = open_ms;
    b.bucket_close_ms = open_ms + period_ms - 1;   // Binance's; the core ignores it
    const double spread = px * 0.006 * (0.3 + w.next());
    b.open = px;
    b.high = px + spread;
    b.low = px - spread;
    b.close = px + spread * (w.next() - 0.5) * 1.6;
    b.volume = 100.0 + 900.0 * w.next();
    b.quote_volume = b.volume * b.close;
    b.number_of_trades = static_cast<int64_t>(500 + 5000 * w.next());
    b.taker_buy_base_volume = b.volume * (0.2 + 0.6 * w.next());
    b.taker_buy_quote_volume = b.taker_buy_base_volume * b.close;
    b.from_backfill = true;
    return b;
}

// n closed bars of period `period_ms`, the NEWEST closing at `last_close_ms`.
std::vector<KlineBar> series(int n, int64_t period_ms, int64_t last_close_ms, uint64_t seed)
{
    Walk w(seed);
    double px = 100.0;
    std::vector<KlineBar> out;
    const int64_t first_open = last_close_ms - static_cast<int64_t>(n) * period_ms;
    for (int i = 0; i < n; ++i) out.push_back(makeBar(first_open + i * period_ms, period_ms, px, w));
    return out;
}

void tick(ICore& core, int64_t ts_ms, double px, uint64_t& tid)
{
    TickEvent ev{};
    ev.product_id = kProduct;
    ev.exchange_ts_ns = static_cast<uint64_t>(ts_ms) * 1'000'000ULL;
    ev.recv_ts_ns = ev.exchange_ts_ns + 500;
    ev.bid_px = px - 0.05;
    ev.ask_px = px + 0.05;
    ev.has_book = true;
    ev.last_px = px;
    ev.last_qty = 1.0;
    ev.has_trade = true;
    ev.last_trade_ts_ms = static_cast<uint64_t>(ts_ms);
    ev.last_trade_id = ++tid;
    ev.aggressor_is_buy = static_cast<int>(tid & 1ULL);
    ev.update_kind = 6;
    core.onTick(ev);
}

int drain(ICore& core, int64_t* last_open)
{
    int n = 0;
    KlineBar b{};
    while (core.barReady(&b)) {
        ++n;
        if (last_open) *last_open = b.bucket_open_ms;
    }
    return n;
}

void writeText(const std::string& path, const std::string& text)
{
    std::ofstream f(path.c_str());
    if (!f) throw std::runtime_error("cannot write " + path);
    f << text;
}

// One tiny linear model per directory; the predictions are not under test here,
// only that the PREFIXED directory names load.
void writeWeights(const std::string& root, const std::vector<std::string>& dirs)
{
    ::mkdir(root.c_str(), 0755);
    for (const std::string& d : dirs) {
        const std::string p = root + "/" + d;
        ::mkdir(p.c_str(), 0755);
        writeText(p + "/model.txt", "model_kind linear\nformat_version 1\nn_features 1\n"
                                    "intercept 0.5\ncoef\n1\n");
        writeText(p + "/scaler.txt", "1\n0 1\n");
        writeText(p + "/features.txt", "close\n");
    }
}

// The independent expectation for one context timeframe.
struct Expect {
    Table panel;
    std::vector<int64_t> close_ms;
    int64_t tf_ms{0};
};

Expect expectFor(const std::vector<KlineBar>& ingested, int64_t tf_ms, int64_t now_ms)
{
    std::vector<KlineBar> closed;
    for (const KlineBar& b : ingested)
        if (b.bucket_open_ms + tf_ms <= now_ms) closed.push_back(b);
    if (closed.size() > PANEL_BARS)
        closed.erase(closed.begin(), closed.end() - static_cast<std::ptrdiff_t>(PANEL_BARS));
    RawBars rb;
    for (const KlineBar& b : closed) {
        rb.open.push_back(b.open);
        rb.high.push_back(b.high);
        rb.low.push_back(b.low);
        rb.close.push_back(b.close);
        rb.volume.push_back(b.volume);
        rb.quote_volume.push_back(b.quote_volume);
        rb.taker_buy_quote_volume.push_back(b.taker_buy_quote_volume);
        rb.number_of_trades.push_back(static_cast<double>(b.number_of_trades));
    }
    Expect e;
    e.panel = engineerFeaturesContext(rb);
    e.tf_ms = tf_ms;
    for (const KlineBar& b : closed) e.close_ms.push_back(b.bucket_open_ms + tf_ms);
    return e;
}

// The as-of row, computed here rather than borrowed from the core.
int asofRow(const Expect& e, int64_t base_open_ms)
{
    const int64_t need = (base_open_ms / e.tf_ms) * e.tf_ms;
    for (int i = static_cast<int>(e.close_ms.size()) - 1; i >= 0; --i) {
        if (e.close_ms[static_cast<size_t>(i)] <= base_open_ms)
            return e.close_ms[static_cast<size_t>(i)] == need ? i : -1;
    }
    return -1;
}

bool atomAt(const Expect& e, uint16_t code, Position pos, int row)
{
    if (row < 0) return false;
    if (!positionAllowed({code}, pos)) return false;
    return atomMask(e.panel, code, pos).at(static_cast<size_t>(row)) != 0;
}

struct Leg {
    std::string dir;
    RegimeSpec spec{};
};

Leg leg(const std::vector<uint16_t>& codes, const std::vector<uint32_t>& tfs, Position pos)
{
    Leg l;
    l.dir = regimeDirName(codes, tfs, pos);
    l.spec.n_atoms = static_cast<uint8_t>(codes.size());
    l.spec.position = static_cast<int8_t>(pos);
    for (size_t k = 0; k < codes.size(); ++k) {
        l.spec.atom_codes[k] = codes[k];
        l.spec.atom_tf_sec[k] = tfs[k];
    }
    return l;
}

bool throws(const std::function<void()>& f)
{
    try { f(); } catch (const std::invalid_argument&) { return true; }
    return false;
}

} // namespace

int main()
{
    std::printf("=== ABI 6: context timeframes (orb) ===\n");
    DecisionGate gate{};
    gate.threshold_long = 0.0007;
    gate.threshold_short = 0.0013;
    gate.threshold_center_long = 0.0;
    gate.threshold_center_short = 0.0;
    gate.reverse = 1;

    std::vector<uint16_t> known;
    for (unsigned c = 0; c < 4096; ++c)
        if (atomIsKnown(static_cast<uint16_t>(c))) known.push_back(static_cast<uint16_t>(c));
    check(known.size() >= 20, "the gate knows at least 20 atom codes");
    const uint32_t k15 = 900, k1h = 3600, k1d = 86400;
    const int ctx[] = {3600, 86400};

    // The stack: every known atom as a single-atom 1h, 1d and 15m regime on
    // both positions, plus mixed conjunctions over the first few codes.
    std::vector<Leg> legs;
    for (const uint16_t c : known)
        for (const Position p : {Position::LONG, Position::SHORT}) {
            legs.push_back(leg({c}, {k1h}, p));
            legs.push_back(leg({c}, {k1d}, p));
            legs.push_back(leg({c}, {k15}, p));
        }
    const size_t n_single = legs.size();
    struct Mixed { size_t idx; uint16_t a, b, c; Position p; };
    std::vector<Mixed> mixed;
    for (size_t i = 0; i + 2 < known.size() && mixed.size() < 12; i += 2)
        for (const Position p : {Position::LONG, Position::SHORT}) {
            mixed.push_back({legs.size(), known[i], known[i + 1], known[i + 2], p});
            legs.push_back(leg({known[i], known[i + 1], known[i + 2]}, {k15, k1h, k1d}, p));
        }

    const std::string weights = "/tmp/agamotto_ctx_tf_weights";
    {
        std::vector<std::string> dirs;
        for (const Leg& l : legs) dirs.push_back(l.dir);
        // The 4h leg [1] must be refused BY ITS TIMEFRAME, so it gets a weights
        // directory: without one the model load would refuse it instead and
        // the timeframe check would be untested (a mutant survived that way).
        dirs.push_back(leg({known[0]}, {14400}, Position::LONG).dir);
        writeWeights(weights, dirs);
    }
    std::vector<RegimeSpec> specs;
    for (const Leg& l : legs) specs.push_back(l.spec);

    // ----------------------------------------------------------------- [0]
    std::printf("[0] createCore validates the context timeframes\n");
    {
        const int eq[] = {900};
        const int off[] = {3601};
        const int dup[] = {3600, 3600};
        const int nine[] = {3600, 7200, 10800, 14400, 18000, 21600, 25200, 28800, 86400};
        check(throws([&] { createCore(kProduct, kBarSec, kWarmup, weights.c_str(), gate, eq, 1); }),
              "a context timeframe equal to bar_sec is refused");
        check(throws([&] { createCore(kProduct, kBarSec, kWarmup, weights.c_str(), gate, off, 1); }),
              "a context timeframe off the minute grid is refused");
        check(throws([&] { createCore(kProduct, kBarSec, kWarmup, weights.c_str(), gate, dup, 2); }),
              "a repeated context timeframe is refused");
        check(throws([&] { createCore(kProduct, kBarSec, kWarmup, weights.c_str(), gate, nine, 9); }),
              "more than 8 context timeframes are refused");
        check(throws([&] { createCore(kProduct, kBarSec, kWarmup, weights.c_str(), gate, nullptr, 1); }),
              "n_context > 0 with a null list is refused");
        auto ok = createCore(kProduct, kBarSec, kWarmup, weights.c_str(), gate, nullptr, 0);
        check(ok && ok->coreAbiVersion() == AGAMOTTO_CORE_ABI_VERSION,
              "n_context = 0 (agamotto) constructs, at this ABI");
        check(ok && ok->coreBuildTag().find("-ctxtf") != std::string::npos,
              "the build tag names the context-timeframe build");
        ContextStats st{};
        check(ok && !ok->contextStats(3600, &st), "contextStats is false for an unconfigured timeframe");
    }

    auto core = createCore(kProduct, kBarSec, kWarmup, weights.c_str(), gate, ctx, 2);

    // ----------------------------------------------------------------- [1]
    std::printf("[1] setRegimeStack validates timeframes and loads PREFIXED directories\n");
    {
        Leg bad = leg({known[0]}, {14400}, Position::LONG);   // 4h: not configured
        std::string why;
        try { core->setRegimeStack(&bad.spec, 1); } catch (const std::invalid_argument& e) { why = e.what(); }
        check(why.find("context timeframe") != std::string::npos,
              "an atom on an unconfigured timeframe is refused FOR ITS TIMEFRAME");
        Leg mix = leg({known[0], known[1]}, {k1h, 0}, Position::LONG);
        check(throws([&] { core->setRegimeStack(&mix.spec, 1); }),
              "a regime mixing prefixed and bare atoms is refused");
        Leg slot = leg({known[0]}, {k1h}, Position::LONG);
        slot.spec.atom_tf_sec[3] = 3600;
        check(throws([&] { core->setRegimeStack(&slot.spec, 1); }),
              "a timeframe in an unused slot is refused");
        check(regimeDirName({12, 3}, {86400, 900}, Position::LONG) == "1d_r012_and_15m_r003_long",
              "regimeDirName restores each atom's prefix, in order");
        check(timeframeLabel(14400) == "4h" && timeframeLabel(300) == "5m",
              "timeframeLabel spells 4h and 5m");
        core->setRegimeStack(specs.data(), static_cast<int>(specs.size()));
        check(core->regimeStackSize() == static_cast<int>(specs.size()),
              "the whole stack installed (" + std::to_string(specs.size()) + " regimes)");
    }

    // ----------------------------------------------------------------- [2]
    // The base timeline. PANEL_BARS backfill from kBase; the seam bucket B is
    // dropped as a partial and repaired, exactly as core_integration_driver
    // does; each later tick pops the previous bucket.
    const int64_t B = kBase + static_cast<int64_t>(PANEL_BARS) * kP;
    // Scored bars: B+2, B+3, ... (see the sequence below). B+2 opens at
    // kBase + 801 * 15m = kBase + 200h 15m.
    const int64_t T1 = B + 2 * kP;
    const int64_t H0 = (T1 / kH) * kH;                 // the hour T1 needs
    const int64_t D0 = (T1 / kD) * kD;                 // the day T1 needs

    const int64_t now_ms = static_cast<int64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    std::vector<KlineBar> h1 = series(300, kH, H0, 11);
    std::vector<KlineBar> d1 = series(150, kD, D0, 13);   // SHORT: a young symbol

    std::printf("[2] ingestContextBars validation\n");
    {
        check(!core->ingestContextBars(7200, h1.data(), static_cast<int>(h1.size())),
              "an unconfigured timeframe is refused");
        std::vector<KlineBar> off = h1;
        off[5].bucket_open_ms += 60000;
        check(!core->ingestContextBars(3600, off.data(), static_cast<int>(off.size())),
              "an off-grid bar is refused");
        std::vector<KlineBar> back = h1;
        std::swap(back[10], back[11]);
        check(!core->ingestContextBars(3600, back.data(), static_cast<int>(back.size())),
              "a non-increasing series is refused");
        ContextStats st{};
        core->contextStats(3600, &st);
        check(st.ingests == 0 && st.ingests_refused == 2 && st.bars_held == 0,
              "refused ingests changed nothing and were counted");

        std::vector<KlineBar> big = series(900, kH, H0, 17);
        check(core->ingestContextBars(3600, big.data(), static_cast<int>(big.size())),
              "900 bars accepted");
        core->contextStats(3600, &st);
        check(st.bars_held == static_cast<int64_t>(PANEL_BARS), "at most PANEL_BARS are held");

        // A still-open bar (its close is in the future) is dropped by TIME.
        std::vector<KlineBar> withOpen = h1;
        KlineBar fut = h1.back();
        fut.bucket_open_ms = ((now_ms / kH) + 24) * kH;
        withOpen.push_back(fut);
        check(core->ingestContextBars(3600, withOpen.data(), static_cast<int>(withOpen.size())),
              "a series ending in a still-open bar is accepted");
        core->contextStats(3600, &st);
        check(st.open_bars_dropped == 1 && st.bars_held == 300,
              "the still-open bar was dropped, the 300 closed ones held");
        check(st.newest_close_ms == H0, "newest_close_ms = open + tf of the newest closed bar");
        check(st.gaps_last_ingest == 1, "the jump to the future bar is counted as one gap");

        check(core->ingestContextBars(86400, d1.data(), static_cast<int>(d1.size())),
              "a SHORT 1d history (150 bars) is accepted");
    }

    // Warm the base and reach the first scored bar.
    uint64_t tid = 0;
    Walk w15(3);
    double px15 = 64000.0;
    {
        std::vector<KlineBar> bf;
        for (int i = 0; i < static_cast<int>(PANEL_BARS); ++i)
            bf.push_back(makeBar(kBase + i * kP, kP, px15, w15));
        check(core->ingestBackfill(bf.data(), static_cast<int>(bf.size())), "base backfill accepted");
        tick(*core, B + 300000, px15, tid);
        tick(*core, B + kP + 10000, px15, tid);
        tick(*core, B + 2 * kP + 10000, px15, tid);
        drain(*core, nullptr);
        KlineBar fill = makeBar(B, kP, px15, w15);
        check(core->ingestBackfill(&fill, 1), "seam repaired");
        check(core->isWarm(), "base warm");
    }
    int64_t next_tick_bucket = B + 3 * kP;
    auto scoreNext = [&]() -> int64_t {
        tick(*core, next_tick_bucket + 10000, px15, tid);
        next_tick_bucket += kP;
        int64_t open = 0;
        drain(*core, &open);
        return open;
    };

    // Grade one scored bar: every single-atom context regime against the
    // independent expectation. Returns how many fired, so "0 fired everywhere"
    // cannot pass silently.
    std::map<size_t, bool> base_fired;   // base-only regime index -> fired
    auto grade = [&](int64_t T, const Expect& eh, const Expect& ed, const char* tag,
                     bool h_expected_usable) {
        const int rh = asofRow(eh, T);
        const int rd = asofRow(ed, T);
        check((rh >= 0) == h_expected_usable, std::string(tag) + ": 1h as-of usability");
        check(rd >= 0, std::string(tag) + ": 1d as-of row is usable");
        int fired = 0, mism = 0;
        base_fired.clear();
        for (size_t i = 0; i < n_single; ++i) {
            const RegimeSpec& s = legs[i].spec;
            const Position p = s.position > 0 ? Position::LONG : Position::SHORT;
            const bool got = core->regimeFiredLatest(static_cast<int>(i));
            if (s.atom_tf_sec[0] == k15) { base_fired[i] = got; continue; }
            const bool want = (s.atom_tf_sec[0] == k1h) ? atomAt(eh, s.atom_codes[0], p, rh)
                                                        : atomAt(ed, s.atom_codes[0], p, rd);
            if (got != want) ++mism;
            if (got) ++fired;
        }
        check(mism == 0, std::string(tag) + ": every single-atom 1h/1d regime matches the "
                         "independent as-of expectation (" + std::to_string(mism) + " mismatches)");
        // Mixed: AND of the base-only regime's flag and the two context atoms.
        int mm = 0;
        for (const Mixed& m : mixed) {
            size_t base_idx = 0;
            for (size_t i = 0; i < n_single; ++i) {
                const RegimeSpec& s = legs[i].spec;
                if (s.atom_tf_sec[0] == k15 && s.atom_codes[0] == m.a &&
                    s.position == static_cast<int8_t>(m.p)) { base_idx = i; break; }
            }
            const bool allowed = positionAllowed({m.a, m.b, m.c}, m.p);
            const bool want = allowed && base_fired[base_idx] && atomAt(eh, m.b, m.p, rh) &&
                              atomAt(ed, m.c, m.p, rd);
            if (core->regimeFiredLatest(static_cast<int>(m.idx)) != want) ++mm;
        }
        check(mm == 0, std::string(tag) + ": every mixed 15m/1h/1d conjunction is the AND of "
                       "its parts (" + std::to_string(mm) + " mismatches)");
        return fired;
    };

    // Sensitivity: the sweep must be able to SEE a wrong row and a wrong panel,
    // or matching it proves nothing.
    auto sensitive = [&](const Expect& e, int row, int other_row, const Expect* other_panel) {
        int diff = 0;
        for (const uint16_t c : known)
            for (const Position p : {Position::LONG, Position::SHORT}) {
                const bool a = atomAt(e, c, p, row);
                const bool b = other_panel ? atomAt(*other_panel, c, p,
                                                    static_cast<int>(other_panel->close_ms.size()) - 1)
                                           : atomAt(e, c, p, other_row);
                if (a != b) ++diff;
            }
        return diff;
    };

    // ----------------------------------------------------------------- [3]
    std::printf("[3] fresh lookups: the as-of row, every atom, both positions\n");
    Expect eh = expectFor(h1, kH, now_ms);
    const Expect ed = expectFor(d1, kD, now_ms);
    int total_fired = 0;
    for (int k = 0; k < 2; ++k) {                         // H0+15m, H0+30m
        const int64_t T = scoreNext();
        check(T == T1 + k * kP, "scored the expected base bar");
        total_fired += grade(T, eh, ed, ("fresh T+" + std::to_string(k)).c_str(), true);
    }
    check(total_fired > 0, "some context regimes fired (the sweep is not all-False)");
    check(sensitive(eh, asofRow(eh, T1), asofRow(eh, T1) - 1, nullptr) > 0,
          "a ONE-ROW SHIFT changes some atom (the sweep can see an off-by-one)");
    check(sensitive(eh, asofRow(eh, T1), 0, &ed) > 0,
          "the 1d panel's newest row differs from the 1h as-of row on some atom "
          "(the sweep can see a wrong-timeframe panel)");

    // ----------------------------------------------------------------- [4]
    // The next base bar opens at H0 + 45m, so the 1h bar closing at H0 + 1h
    // closes TOGETHER with it. It is ingested first and must NOT be read.
    std::printf("[4] the one-bar lag: the context bar closing with the base bar is not used\n");
    {
        std::vector<KlineBar> h2 = h1;
        double px = h1.back().close;
        Walk w(23);
        h2.push_back(makeBar(H0, kH, px, w));            // closes at H0 + 1h
        check(core->ingestContextBars(3600, h2.data(), static_cast<int>(h2.size())),
              "the bar closing with the base bar is ingested");
        eh = expectFor(h2, kH, now_ms);
        const int64_t T = scoreNext();
        check(T == H0 + 3 * kP, "the base bar opens at H0 + 45m");
        check(asofRow(eh, T) == static_cast<int>(eh.close_ms.size()) - 2,
              "the as-of row is the SECOND newest, not the newest");
        grade(T, eh, ed, "lag", true);
        check(sensitive(eh, asofRow(eh, T), asofRow(eh, T) + 1, nullptr) > 0,
              "reading the newest bar instead would change some atom");
        ContextStats st{};
        core->contextStats(3600, &st);
        check(st.last_lookup_close_ms == H0, "the lookup used the bar closing at H0");
        check(st.lookups_stale == 0 && st.lookups_missing == 0, "no stale or missing lookups yet");

        // H0 + 1h: the needed bar is now the newest one.
        const int64_t T2 = scoreNext();
        check(T2 == H0 + kH, "the next base bar opens on the hour");
        grade(T2, eh, ed, "on-the-hour", true);
        core->contextStats(3600, &st);
        check(st.last_lookup_close_ms == H0 + kH, "the on-the-hour bar reads the bar closing at it");
    }

    // ----------------------------------------------------------------- [5]
    std::printf("[5] stale: the needed 1h bar is missing -> no 1h regime fires\n");
    {
        for (int k = 0; k < 3; ++k) scoreNext();          // H0+1h15 .. H0+1h45: still fresh
        ContextStats before{};
        core->contextStats(3600, &before);
        const int64_t T = scoreNext();                    // H0 + 2h: needs a bar never ingested
        check(T == H0 + 2 * kH, "the base bar opens at H0 + 2h");
        ContextStats st{};
        core->contextStats(3600, &st);
        check(st.lookups_stale == before.lookups_stale + 1, "exactly one stale lookup counted");
        check(st.last_lookup_close_ms == H0 + kH, "it found only the older bar");
        int h_fired = 0, mixed_fired = 0;
        for (size_t i = 0; i < n_single; ++i)
            if (legs[i].spec.atom_tf_sec[0] == k1h && core->regimeFiredLatest(static_cast<int>(i)))
                ++h_fired;
        for (const Mixed& m : mixed)
            if (core->regimeFiredLatest(static_cast<int>(m.idx))) ++mixed_fired;
        check(h_fired == 0, "NO 1h regime fires on a stale 1h bar");
        check(mixed_fired == 0, "NO conjunction naming 1h fires on a stale 1h bar");
        grade(T, eh, ed, "stale (1d still graded)", false);
        ContextStats sd{};
        core->contextStats(86400, &sd);
        check(sd.lookups_stale == 0 && sd.panel_rows == 150,
              "the 1d timeframe is unaffected, on its 150-row short panel");
    }

    std::printf("\n%s: %d checks, %d failures\n", g_failures ? "FAILED" : "CTX-TF PASS",
                g_checks, g_failures);
    return g_failures ? 1 : 0;
}
