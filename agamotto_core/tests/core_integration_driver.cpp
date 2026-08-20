// STAGE 2.6 INTEGRATION GATE — the whole in-process chain, end to end.
//
// tests/feature_parity.py grades engineerFeatures() against research.py on a
// panel handed to BOTH sides. That is the right gate for the MATH and the wrong
// gate for the WIRING: it never constructs a core, never builds a bar, and
// would pass unchanged if RealCore called the engine on the wrong window, at
// the wrong moment, or not at all. `engineerFeatures` existed and nothing
// called it, and the 65-column harness was green throughout.
//
// This driver closes that hole. It drives SYNTHETIC TICKS through the real
// createCore() -> KlineBuilder -> engineerFeatures path and asserts the four
// properties the parity harness structurally cannot see:
//
//   1. NO PANEL BEFORE WARM. Every bar popped while the contiguous run is
//      short must leave panel_rows == 0 and panels_computed == 0. A core that
//      engineered a short panel would be caught by engineerFeatures' width
//      check; a core that engineered a panel it should not have YET would not.
//   2. EXACTLY 699 x 65 ON THE FIRST WARM BAR — the width live engineers
//      (trading.py fetches 700 and drops the incomplete one), not the 700 the
//      ring retains, and the full declared column set.
//   3. THE PANEL IS STAMPED WITH THE BAR THAT WAS POPPED, so a Phase-3 gate
//      cannot score a panel that describes a different bucket.
//   4. A BURST DOES NOT PRODUCE STALE PANELS. Flat bars drained from one tick
//      already have successors in history; the panel ends at the NEWEST bar, so
//      the older ones must be SKIPPED and COUNTED, never paired with it. That
//      pairing is lookahead and would flatter, not fail, a Phase-4 backtest.
//   5. PHASE 4: THE MODELS ARE WIRED TO THE GATE AND TO THE PANEL. A firing
//      regime must produce a y_pred computed from the NEWEST row of the columns
//      ITS OWN features.txt names; a non-firing one must produce NaN. Graded
//      against a closed-form expectation the driver computes from the ABI's own
//      panelLatest() accessor — a different code path from the core's internal
//      one — so a model reading the wrong row, the wrong column, or ANOTHER
//      regime's weights is caught here rather than only in a number nobody
//      checks.
//
// THE WEIGHTS ARE SYNTHESIZED, NOT THE DEPLOYED ONES, AND THAT IS THE POINT.
// tests/model_parity.py grades the ARITHMETIC against the deployed sklearn
// pipeline on the real export. This file grades the WIRING, so it writes its
// own tiny weight tree with coefficients it chose: an expectation it can state
// exactly beats an expectation it has to trust. It also keeps the gate
// hermetic, which is why build_linux.sh can run it inside the build image with
// no weights tree mounted. `--weights DIR` overrides it for a smoke test
// against a real export.
//
// The run also reports the MEASURED per-bar cost of the panel on this exact
// path. Phase 1's budget note was an estimate assembled from a TA-Lib
// microbenchmark; this is the thing itself.
//
//   ./core_integration_driver            # gate: asserts, exits nonzero on any failure
//   ./core_integration_driver --bench N  # additionally time N consecutive live panels
#include "agamotto_core.hpp"
#include "codes_generated.hpp"
#include "feature_engine.hpp"

#include <sys/stat.h>

#include <algorithm>
#include <cinttypes>
#include <fstream>
#include <cmath>
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

template <typename T>
void checkEq(T got, T want, const std::string& what)
{
    ++g_checks;
    if (got != want) {
        ++g_failures;
        std::printf("  FAIL  %s: got %lld, want %lld\n", what.c_str(),
                    static_cast<long long>(got), static_cast<long long>(want));
    }
}

constexpr int      kBarSec  = 900;               // agamotto 15m
constexpr int      kWarmup  = 700;               // the contract's warmup
constexpr int64_t  kPeriodMs = kBarSec * 1000LL;
constexpr int64_t  kProduct = 2000000170LL;
// An on-grid bucket well inside the plausible-timestamp window the builder
// enforces. 2026-01-01T00:00:00Z rounded onto the 900s grid.
constexpr int64_t  kBase    = 1767225600000LL;

// A deterministic price walk. NOT std::rand: the gate must reproduce bit for
// bit on every machine, and a libc-dependent generator makes a failure
// unreproducible on the host that did not see it.
struct Walk {
    uint64_t s{0x9E3779B97F4A7C15ULL};
    double next()
    {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        return static_cast<double>(s >> 11) / static_cast<double>(1ULL << 53);
    }
};

// One synthetic CLOSED bar, in the shape the builder emits and the backfill
// parser produces. Prices are kept well away from a constant series: a panel of
// identical closes drives several indicators into their constant-window guards
// and would grade the engine on its degenerate branches only.
KlineBar makeBar(int64_t bucket_open_ms, double px, Walk& w)
{
    KlineBar b{};
    b.bucket_open_ms = bucket_open_ms;
    b.bucket_close_ms = bucket_open_ms + kPeriodMs - 1;
    const double spread = px * 0.0015 * (0.5 + w.next());
    b.open = px;
    b.high = px + spread;
    b.low = px - spread;
    b.close = px + spread * (w.next() - 0.5);
    b.volume = 100.0 + 900.0 * w.next();
    b.quote_volume = b.volume * b.close;
    b.number_of_trades = static_cast<int64_t>(500 + 5000 * w.next());
    b.taker_buy_base_volume = b.volume * (0.3 + 0.4 * w.next());
    b.taker_buy_quote_volume = b.taker_buy_base_volume * b.close;
    b.aggressor_source = KlineBar::AggressorSource::EXACT_MAKER_FLAG;
    b.from_backfill = true;
    return b;
}

// Feed one trade into the core, in the bucket that contains ts_ms.
void tick(ICore& core, int64_t ts_ms, double px, double qty, uint64_t& trade_id)
{
    TickEvent ev{};
    ev.product_id = kProduct;
    ev.exchange_ts_ns = static_cast<uint64_t>(ts_ms) * 1'000'000ULL;
    ev.recv_ts_ns = ev.exchange_ts_ns + 500;
    ev.bid_px = px - 0.05;
    ev.ask_px = px + 0.05;
    ev.has_book = true;
    ev.last_px = px;
    ev.last_qty = qty;
    ev.has_trade = true;
    ev.last_trade_ts_ms = static_cast<uint64_t>(ts_ms);
    ev.last_trade_id = ++trade_id;
    ev.aggressor_is_buy = static_cast<int>(trade_id & 1ULL);
    ev.update_kind = 6;    // aef UpdateKind::TRADE_UPDATE — the ONLY kind that
                           // contributes volume (rule 1).
    core.onTick(ev);
}

// ---------------------------------------------------------------------------
// THE SYNTHETIC WEIGHT TREE.
//
// One directory per regime, in the exact three-file format
// marvel/gauntlet/export_agamotto_sentinel_weights.py writes, with
// coefficients chosen here so the expected prediction is a closed form.
//
// The feature COUNTS deliberately differ per regime (1, 2, 3, ...). The
// deployed window mixes 5-feature and 16-feature models under one label, so a
// runner that assumed a single width would be wrong on live data; a fixture
// that used one width everywhere could not tell.
//
// The COEFFICIENTS deliberately differ per regime too, so a core that handed
// regime i the model of regime j fails rather than predicting a plausible
// number from the wrong weights.
// ---------------------------------------------------------------------------
struct SynthModel {
    std::vector<std::string> features;
    std::vector<double> center, scale, coef;
    double intercept{0.0};
};

// Panel columns the synthetic models read. All three are emitted by
// engineerFeatures and are finite on the clean walk this driver builds.
// Referenced through codes_generated.hpp so a renamed code breaks the build
// rather than the expectation.
const char* const kSynthFeatureCodes[] = {"close", codes::F_PRICE_RANGE_PCT,
                                          codes::F_HIGH_OPEN_PCT};

SynthModel synthModelFor(size_t i)
{
    SynthModel m;
    const size_t n = (i % 3) + 1;          // 1, 2 or 3 features
    for (size_t k = 0; k < n; ++k) {
        m.features.push_back(kSynthFeatureCodes[k]);
        m.center.push_back(1000.0 * static_cast<double>(k + 1)
                           + 7.0 * static_cast<double>(i));
        m.scale.push_back(2.0 + static_cast<double>(k) + 0.25 * static_cast<double>(i));
        m.coef.push_back(3.0 - 0.5 * static_cast<double>(k)
                         + 1.5 * static_cast<double>(i));
    }
    m.intercept = 0.125 * static_cast<double>(i + 1);
    return m;
}

void writeText(const std::string& path, const std::string& text)
{
    std::ofstream f(path.c_str());
    if (!f) throw std::runtime_error("cannot write " + path);
    f << text;
}

std::string fmtG17(double v)
{
    char b[64];
    std::snprintf(b, sizeof(b), "%.17g", v);
    return b;
}

void writeSynthWeights(const std::string& root, const std::vector<std::string>& dirs)
{
    if (::mkdir(root.c_str(), 0755) != 0) {
        struct stat st;
        if (::stat(root.c_str(), &st) != 0 || !S_ISDIR(st.st_mode))
            throw std::runtime_error("cannot create " + root);
    }
    for (size_t i = 0; i < dirs.size(); ++i) {
        const std::string d = root + "/" + dirs[i];
        ::mkdir(d.c_str(), 0755);
        const SynthModel m = synthModelFor(i);
        std::string model = "model_kind linear\nformat_version 1\nn_features "
                          + std::to_string(m.features.size()) + "\nintercept "
                          + fmtG17(m.intercept) + "\ncoef\n";
        for (double c : m.coef) model += fmtG17(c) + "\n";
        writeText(d + "/model.txt", model);
        std::string sc = std::to_string(m.features.size()) + "\n";
        for (size_t k = 0; k < m.features.size(); ++k)
            sc += fmtG17(m.center[k]) + " " + fmtG17(m.scale[k]) + "\n";
        writeText(d + "/scaler.txt", sc);
        std::string ft;
        for (const std::string& f : m.features) ft += f + "\n";
        writeText(d + "/features.txt", ft);
    }
}

// Drain the ready queue, returning the bars in order. The panel is computed
// inside barReady(), so this is also what drives it.
std::vector<KlineBar> drain(ICore& core)
{
    std::vector<KlineBar> out;
    KlineBar b{};
    while (core.barReady(&b)) out.push_back(b);
    return out;
}

double pct(std::vector<int64_t> v, double p)
{
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t i = static_cast<size_t>(p * (v.size() - 1) + 0.5);
    return static_cast<double>(v[std::min(i, v.size() - 1)]);
}

// THE CLOSED FORM, for every FIRING regime on the panel the core currently
// holds. Recomputed from panelLatest() -- the ABI's own accessor, a DIFFERENT
// code path from the one the core predicted through -- so a model that read the
// wrong ROW, the wrong COLUMN, or ANOTHER regime's coefficients is caught by a
// number rather than by a shape.
//
// Returns how many regimes it actually graded, because "0 fired" must never be
// mistaken for "the check passed": the caller asserts the run-wide total.
int checkClosedForm(const ICore& core, int n_stack)
{
    const CoreDiagnostics d = core.diagnostics();
    int graded = 0;
    for (int i = 0; i < n_stack; ++i) {
        if (!core.regimeFiredLatest(i)) continue;
        const SynthModel m = synthModelFor(static_cast<size_t>(i));
        double want = m.intercept;
        for (size_t k = 0; k < m.features.size(); ++k) {
            int col = -1;
            for (int j = 0; j < static_cast<int>(d.panel_cols); ++j) {
                const char* c = core.panelColumnCode(j);
                if (c != nullptr && m.features[k] == c) { col = j; break; }
            }
            check(col >= 0, "the model's feature is a panel column");
            double x = (col >= 0) ? core.panelLatest(col) : 0.0;
            // trading.py:700's fill, mirrored so the expectation stays valid if
            // a column happens to be NaN on this walk.
            if (std::isnan(x)) x = 0.0;
            want += m.coef[k] * ((x - m.center[k]) / m.scale[k]);
        }
        const double got = core.regimePrediction(i);
        const double denom = std::max(std::fabs(want), 1.0);
        check(std::fabs(got - want) / denom < 1e-12,
              "the prediction equals intercept + sum coef*((x-center)/scale) on "
              "the panel's NEWEST row, with THIS regime's own weights");
        ++graded;
    }
    return graded;
}

int run(int bench_n, const std::string& weights_override,
        const std::string& tmp_root)
{
    std::printf("=== stage 2.6 integration: ticks -> KlineBuilder -> RealCore -> engineerFeatures ===\n");
    std::printf("bar_sec=%d warmup_bars=%d PANEL_BARS=%zu\n", kBarSec, kWarmup, PANEL_BARS);

    // ----------------------------------------------------------------- (0)
    // A warmup that cannot slice the panel must be refused AT CONSTRUCTION.
    // The alternative — a core that returns "not warm" forever while reporting
    // warm=1 on every bar line — is the exact failure mode this whole file
    // exists to make impossible.
    // The six regimes this run installs, as the CODED directory names the
    // exporter uses. Written out FIRST so createCore has something real to
    // validate and setRegimeStack has something real to load.
    const std::vector<std::string> kWeightDirs = {
        "r029_and_r045_long", "r069_and_r065_short", "r039_and_r008_long",
        "r060_and_r075_long", "r029_and_r066_and_r073_long",
        "r069_and_r040_and_r074_short",
    };
    const std::string weights_ =
        weights_override.empty() ? (tmp_root + "/agamotto_integration_weights")
                                 : weights_override;
    if (weights_override.empty()) {
        writeSynthWeights(weights_, kWeightDirs);
        std::printf("     synthetic weights at %s (%zu regimes, feature counts 1..3)\n",
                    weights_.c_str(), kWeightDirs.size());
    } else {
        std::printf("     weights OVERRIDE: %s -- the closed-form prediction "
                    "checks are SKIPPED (they only hold for the fixture)\n",
                    weights_.c_str());
    }

    // PHASE 5. The DEPLOYED gate of pred_agamotto.base.15m_1, so this file
    // exercises the numbers production runs rather than round ones that would
    // hide a sign or a leg swap.
    DecisionGate kGate{};
    kGate.threshold_long = 0.00070083;
    kGate.threshold_short = 0.0013514;
    kGate.threshold_center_long = -0.00012442;
    kGate.threshold_center_short = 0.00030606;
    kGate.reverse = 1;

    std::printf("\n[0] createCore refuses a warmup it could never slice\n");
    bool threw_ = false;
    try {
        (void)createCore(kProduct, kBarSec, static_cast<int>(PANEL_BARS),
                         weights_.c_str(), kGate);
    } catch (const std::invalid_argument&) {
        threw_ = true;
    }
    check(threw_, "createCore(warmup=PANEL_BARS) must throw std::invalid_argument");

    // PHASE 4. weights_dir is a REQUIRED key with no default. Each of these is
    // a way a core could boot with no models, gate bars, predict nothing, and
    // produce a silent no-signal day that looks exactly like a working one.
    for (const char* bad_ : {static_cast<const char*>(nullptr), "", "/nonexistent/weights"}) {
        bool rejected_ = false;
        try {
            (void)createCore(kProduct, kBarSec, kWarmup, bad_, kGate);
        } catch (const std::invalid_argument&) {
            rejected_ = true;
        }
        check(rejected_, std::string("createCore REFUSES weights_dir='")
                           + (bad_ ? bad_ : "(null)") + "'");
    }

    // PHASE 5. Every one of these is a gate that would otherwise boot clean and
    // decide. The sub-floor case is the CLAUDE.md hard rule, and it must be
    // REFUSED rather than raised to the floor: an operator who typed 2e-5 must
    // not end up with a strategy silently running at 2e-4.
    {
        struct BadGate { DecisionGate g; const char* what; };
        std::vector<BadGate> bad_;
        DecisionGate g_;
        g_ = kGate; g_.threshold_long = 0.00019;
        bad_.push_back({g_, "threshold_long BELOW the 2 bps floor"});
        g_ = kGate; g_.threshold_short = 0.0;
        bad_.push_back({g_, "threshold_short == 0 (the always-on gate)"});
        g_ = kGate; g_.threshold_long = -0.0007;
        bad_.push_back({g_, "a NEGATIVE width"});
        g_ = kGate; g_.threshold_center_short = std::numeric_limits<double>::quiet_NaN();
        bad_.push_back({g_, "a NaN centre"});
        g_ = kGate; g_.reverse = 0;
        bad_.push_back({g_, "reverse == 0 (a permanently flat bot)"});
        g_ = kGate; g_.reverse = 2;
        bad_.push_back({g_, "reverse == 2 (a silent doubling of live size)"});
        for (const BadGate& b : bad_) {
            bool rejected_ = false;
            try {
                (void)createCore(kProduct, kBarSec, kWarmup, weights_.c_str(), b.g);
            } catch (const std::invalid_argument&) {
                rejected_ = true;
            }
            check(rejected_, std::string("createCore REFUSES ") + b.what);
        }
    }

    std::unique_ptr<ICore> core =
        createCore(kProduct, kBarSec, kWarmup, weights_.c_str(), kGate);
    check(core->coreIsRealImplementation(), "core reports itself real");
    const std::string tag_ = core->coreBuildTag();
    check(tag_.find("-phase5-decision") != std::string::npos,
          "build tag says phase5-decision (got: " + tag_ + ")");
    {
        // The core must report the gate it will COMPARE AGAINST, not the one
        // the caller meant to install.
        const DecisionGate g_ = core->decisionGate();
        checkEq<double>(g_.threshold_long, kGate.threshold_long,
                        "decisionGate() echoes threshold_long");
        checkEq<double>(g_.threshold_short, kGate.threshold_short,
                        "decisionGate() echoes threshold_short");
        checkEq<double>(g_.threshold_center_long, kGate.threshold_center_long,
                        "decisionGate() echoes threshold_center_long");
        checkEq<double>(g_.threshold_center_short, kGate.threshold_center_short,
                        "decisionGate() echoes threshold_center_short");
        checkEq<int>(g_.reverse, kGate.reverse, "decisionGate() echoes reverse");
    }
    std::printf("     core=%s\n", tag_.c_str());
    checkEq<int64_t>(core->diagnostics().models_loaded, 0,
                     "constructing a core loads NO models -- the stack decides which");
    check(std::isnan(core->regimePrediction(0)),
          "no prediction exists before a stack is installed");
    checkEq<int>(core->winningRegimeIndex(), -1,
                 "and no regime has won anything yet");

    // ----------------------------------------------------------------- (0b)
    // STAGE 3: install the regime stack, CODES ONLY. A slice of the deployed
    // pred_agamotto.base.15m_1 stack — three that CAN fire and three that
    // cannot, so both halves of the pinned property are exercised on the LIVE
    // path and not only in tests/regime_parity.py (which never constructs a
    // core and would pass unchanged if RealCore never called the gate).
    std::printf("\n[0b] the regime stack is CODES ONLY and is validated eagerly\n");
    const RegimeSpec kStack[] = {
        // firable: no r073/r074/r075 atom.
        {{29, 45},     2, +1},   // r029_and_r045_long
        {{69, 65},     2, -1},   // r069_and_r065_short
        {{39, 8},      2, +1},   // r039_and_r008_long
        // INERT: each carries a vol-quantile atom whose cutoff column is
        // all-NaN on a 699-row panel. These must never fire — PR #532.
        {{60, 75},     2, +1},   // r060_and_r075_long
        {{29, 66, 73}, 3, +1},   // r029_and_r066_and_r073_long
        {{69, 40, 74}, 3, -1},   // r069_and_r040_and_r074_short
    };
    constexpr int kNStack = static_cast<int>(sizeof(kStack) / sizeof(kStack[0]));
    constexpr int kNFirable = 3;   // the first three above

    // The stack is REFUSED, in full, on anything unusable — at configuration
    // time, not on the first warm bar 7.3 days later.
    struct Bad { RegimeSpec spec; const char* what; };
    const Bad kBad[] = {
        {{{0, 0}, 0, +1},   "a regime with ZERO atoms (the always-fire gate)"},
        {{{29, 45}, 2, 0},  "a regime with position 0"},
        {{{29, 45}, 2, +3}, "a regime with position +3"},
        {{{9999, 45}, 2, +1}, "a regime naming an atom this core cannot evaluate"},
    };
    for (const Bad& b : kBad) {
        bool rejected_ = false;
        try {
            core->setRegimeStack(&b.spec, 1);
        } catch (const std::invalid_argument&) {
            rejected_ = true;
        }
        check(rejected_, std::string("setRegimeStack REFUSES ") + b.what);
    }
    bool rejected_empty_ = false;
    try {
        core->setRegimeStack(kStack, 0);
    } catch (const std::invalid_argument&) {
        rejected_empty_ = true;
    }
    check(rejected_empty_, "setRegimeStack REFUSES an empty stack");
    checkEq<int>(core->regimeStackSize(), 0,
                 "no partially-installed stack survives a rejection");

    // PHASE 4. A stack entry with NO weights directory must halt the boot,
    // naming the regime. The Python bot's equivalent raise
    // (FileNotFoundError: Regime folder <name> not found) is what caught a real
    // stack/weights mismatch; skipping the row would install a leg that
    // silently never trades. r002_and_r003_long has no directory in the
    // fixture, and its atoms ARE known predicates -- so this can only fail on
    // the weights, not on validation that already ran above.
    if (weights_override.empty()) {
        std::vector<RegimeSpec> with_orphan_(kStack, kStack + kNStack);
        RegimeSpec orphan_{};
        orphan_.atom_codes[0] = 29; orphan_.atom_codes[1] = 45;
        orphan_.n_atoms = 2; orphan_.position = -1;
        with_orphan_.push_back(orphan_);
        bool rejected_orphan_ = false;
        std::string why_;
        try {
            core->setRegimeStack(with_orphan_.data(),
                                 static_cast<int>(with_orphan_.size()));
        } catch (const std::invalid_argument& e) {
            rejected_orphan_ = true;
            why_ = e.what();
        }
        check(rejected_orphan_,
              "setRegimeStack REFUSES a regime with no weights directory");
        check(why_.find("r029_and_r045_short") != std::string::npos,
              "and the message NAMES the regime (got: " + why_ + ")");
        checkEq<int>(core->regimeStackSize(), 0,
                     "a weights failure leaves NO partially-installed stack");
    }

    core->setRegimeStack(kStack, kNStack);
    checkEq<int>(core->regimeStackSize(), kNStack, "the stack installed");
    checkEq<int64_t>(core->diagnostics().regimes_configured, kNStack,
                     "regimes_configured crosses the ABI");
    checkEq<int64_t>(core->diagnostics().regime_evals, 0,
                     "installing a stack evaluates nothing");
    checkEq<int64_t>(core->diagnostics().models_loaded, kNStack,
                     "one model per regime was loaded WITH the stack");
    if (weights_override.empty()) {
        const CoreDiagnostics dm_ = core->diagnostics();
        // THE MIXED-PROVENANCE DIAGNOSTIC, on the live path. The fixture spans
        // 1..3 features on purpose; the DEPLOYED window spans 5 and 16 under
        // one label, and a core that reported a single width would hide it.
        checkEq<int64_t>(dm_.model_features_min, 1, "the narrowest model is 1 feature");
        checkEq<int64_t>(dm_.model_features_max, 3, "the widest model is 3 features");
        checkEq<int64_t>(dm_.model_feature_count_variants, 3,
                         "3 DISTINCT feature counts are reported -- nothing may "
                         "assume one width");
        checkEq<int64_t>(dm_.predictions_computed, 0,
                         "installing a stack predicts nothing");
        std::printf("     inventory: %s\n", core->modelInventory().c_str());
    }
    check(std::isnan(core->regimePrediction(0)),
          "no prediction exists before the first panel");
    check(!core->regimeFiredLatest(0), "nothing has fired before the first panel");
    check(!core->regimeFiredLatest(kNStack), "an out-of-range regime reads false");
    check(!core->regimeFiredLatest(-1), "a negative index reads false");

    // ----------------------------------------------------------------- (1)
    // Backfill 699 CLOSED bars — one short of warm, exactly as a real boot is
    // before the seam is repaired.
    std::printf("\n[1] backfill 699 closed bars -> contiguous but NOT warm\n");
    Walk w;
    std::vector<KlineBar> bf;
    double px = 64000.0;
    for (int i = 0; i < 699; ++i) {
        px *= 1.0 + (w.next() - 0.5) * 0.004;
        bf.push_back(makeBar(kBase + static_cast<int64_t>(i) * kPeriodMs, px, w));
    }
    check(core->ingestBackfill(bf.data(), static_cast<int>(bf.size())),
          "ingestBackfill accepts 699 contiguous on-grid bars");
    checkEq<int64_t>(core->barsBuffered(), 699, "contiguous after backfill");
    check(!core->isWarm(), "699 < 700 is NOT warm");
    checkEq<int64_t>(core->diagnostics().panels_computed, 0,
                     "backfill alone computes no panel (nothing was popped)");

    // ----------------------------------------------------------------- (2)
    // Attach mid-bucket at 699 (discarded as a partial, rule 3), then trade in
    // 700 and 701 so bar 700 is emitted. That opens the STRUCTURAL boot seam:
    // bucket 699 is in neither half, so the run is one live bar long.
    std::printf("\n[2] attach mid-bucket -> boot seam -> the popped bar is NOT warm\n");
    uint64_t tid = 0;
    const int64_t b699 = kBase + 699 * kPeriodMs;
    tick(*core, b699 + 300000, px, 1.5, tid);                 // partial, dropped
    tick(*core, b699 + kPeriodMs + 10000, px * 1.001, 2.0, tid);
    tick(*core, b699 + 2 * kPeriodMs + 10000, px * 1.002, 1.0, tid);
    std::vector<KlineBar> got = drain(*core);
    checkEq<size_t>(got.size(), 1u, "one bar built across the seam");
    CoreDiagnostics d = core->diagnostics();
    checkEq<int64_t>(d.seam_gaps, 1, "the boot seam is reported as one gap");
    checkEq<int64_t>(d.pending_bars, 699, "the pre-seam backfill is quarantined, not destroyed");
    check(!core->isWarm(), "a 1-bar contiguous run is not warm");
    checkEq<int64_t>(d.panels_computed, 0, "NO panel is computed before warm");
    checkEq<int64_t>(d.panel_rows, 0, "no panel is held");
    checkEq<int64_t>(d.panel_cols, 0, "no panel is held");
    checkEq<int64_t>(d.panels_skipped_not_warm, 1, "the skip is COUNTED, not silent");
    check(core->panelColumnCode(0) == nullptr, "panelColumnCode refuses when no panel is held");
    check(std::isnan(core->panelLatest(0)), "panelLatest returns NaN when no panel is held");

    // ----------------------------------------------------------------- (3)
    // Repair the seam with the one missing bucket, exactly as the strategy
    // does from a refreshed CSV. The quarantine splices back and the run
    // becomes 701 bars — warm.
    std::printf("\n[3] repair the seam -> warm -> the NEXT bar produces the panel\n");
    KlineBar fill = makeBar(b699, px, w);
    check(core->ingestBackfill(&fill, 1), "ingestBackfill accepts exactly the missing bucket");
    // 701 bars are spliced and the ring caps at max_history (= warmup = 700), so
    // the OLDEST is trimmed. Asserted as 700, not 701, because the cap is the
    // designed behaviour: the panel needs 699 and the ring is deliberately one
    // longer, so trimming the 701st loses nothing the engine reads.
    checkEq<int64_t>(core->barsBuffered(), 700, "the quarantine is spliced back, ring capped");
    check(core->isWarm(), "700 >= 700 is warm");
    checkEq<int64_t>(core->diagnostics().panels_computed, 0,
                     "a splice alone computes no panel — only a popped bar does");

    tick(*core, b699 + 3 * kPeriodMs + 10000, px * 1.003, 1.0, tid);
    got = drain(*core);
    checkEq<size_t>(got.size(), 1u, "one bar built after the repair");
    d = core->diagnostics();

    // THE ASSERTION THIS WHOLE FILE IS FOR.
    checkEq<int64_t>(d.panels_computed, 1, "the first warm bar produces exactly one panel");
    checkEq<int64_t>(d.panel_rows, static_cast<int64_t>(PANEL_BARS),
                     "the panel is exactly PANEL_BARS rows");
    checkEq<int64_t>(d.panel_cols, 65, "the panel is exactly 65 columns");
    checkEq<int64_t>(d.panel_errors, 0, "engineerFeatures did not throw");
    checkEq<int64_t>(d.panel_bar_ts_ms, got.front().bucket_open_ms,
                     "the panel is stamped with the bar that was popped");
    check(d.feature_compute_us > 0, "the panel's cost was measured");

    // Every column is addressable, named, and its newest row is readable.
    // A panel that reported 65 columns and handed back a null name on one of
    // them would still pass a shape-only check.
    int named_ = 0;
    for (int j = 0; j < static_cast<int>(d.panel_cols); ++j) {
        const char* c = core->panelColumnCode(j);
        if (c != nullptr && std::strlen(c) > 0) ++named_;
    }
    checkEq<int>(named_, 65, "all 65 columns carry a non-empty code");
    check(core->panelColumnCode(65) == nullptr, "column 65 is out of range");
    check(core->panelColumnCode(-1) == nullptr, "column -1 is out of range");
    check(std::isnan(core->panelLatest(65)), "panelLatest(65) is NaN, not a neighbour");

    // The three vol-quantile cutoffs are ALL-NaN on a 699-row panel
    // (min_periods=700 > 699). Pinned here as well as in the parity harness:
    // this is the live path, and it is the subject of marvel PR #532. If it
    // ever stops being NaN, 53 deployed regimes start firing against models
    // never trained on a firing regime.
    int nan_q_ = 0;
    for (int j = 0; j < static_cast<int>(d.panel_cols); ++j) {
        const char* c = core->panelColumnCode(j);
        if (c == nullptr) continue;
        // The q80/q90/q95 codes, resolved through the same table the engine
        // emits, so this cannot drift out of sync with a renamed code.
        for (const char* q : {codes::F_PRICE_RANGE_PCT_Q80, codes::F_PRICE_RANGE_PCT_Q90,
                              codes::F_PRICE_RANGE_PCT_Q95}) {
            if (std::strcmp(c, q) == 0 && std::isnan(core->panelLatest(j))) ++nan_q_;
        }
    }
    // NAMED BY CODE, not by the real column name. This string is a literal and
    // lands verbatim in the built binary; the artifact audit build_linux.sh
    // runs scans printable runs, and an assertion message is one of the easiest
    // places for a real name to re-enter a compiled object unnoticed.
    checkEq<int>(nan_q_, 3, std::string(codes::F_PRICE_RANGE_PCT_Q80) + "/" +
                            codes::F_PRICE_RANGE_PCT_Q90 + "/" +
                            codes::F_PRICE_RANGE_PCT_Q95 +
                            " are NaN on a 699-row panel (min_periods=700)");

    // ----------------------------------------------------------------- (3b)
    // STAGE 3: the gate RAN, on the panel that was just computed. Every check
    // below would pass unchanged if RealCore never called regimeMask, which is
    // exactly why they are here and not only in the parity harness.
    std::printf("\n[3b] the regime gate ran on the panel, and the r07x legs did NOT fire\n");
    checkEq<int64_t>(d.regime_evals, 1, "the gate was evaluated exactly once");
    checkEq<int64_t>(d.regime_errors, 0, "the gate did not throw");
    checkEq<int64_t>(d.regimes_configured, kNStack, "still the stack we installed");
    check(d.regime_gate_us >= 0, "the gate's cost was measured");
    for (int i = kNFirable; i < kNStack; ++i) {
        check(!core->regimeFiredLatest(i),
              "an r07x-gated regime CANNOT fire: its cutoff column is all-NaN "
              "(marvel PR #532)");
        checkEq<int64_t>(core->regimeFireCount(i), 0,
                         "and its run count stays at zero");
    }
    check(core->regimeFireCount(kNStack) == 0, "an out-of-range fire count is 0");

    // ----------------------------------------------------------------- (3c)
    // PHASE 4: THE MODELS RAN, ON THE SAME PANEL AND THE SAME ROW THE GATE
    // CLASSIFIED. Every assertion here would pass unchanged if RealCore never
    // called a model -- which is exactly why they are here and not only in
    // tests/model_parity.py, which never constructs a core.
    std::printf("\n[3c] the models ran on the FIRING regimes, and only on those\n");
    {
        int firing_ = 0;
        for (int i = 0; i < kNStack; ++i) if (core->regimeFiredLatest(i)) ++firing_;
        checkEq<int64_t>(d.predictions_computed, static_cast<int64_t>(firing_),
                         "exactly one prediction per FIRING regime -- a gated-out "
                         "bar is never scored");
        checkEq<int64_t>(d.model_errors, 0, "no model threw");
        if (d.model_errors > 0) {
            std::printf("     last model error: %s\n", core->lastModelError().c_str());
        }

        for (int i = 0; i < kNStack; ++i) {
            const double y_ = core->regimePrediction(i);
            if (!core->regimeFiredLatest(i)) {
                // NaN, never 0.0: 0.0 is a legitimate prediction and would read
                // as a confident flat call on a regime that never ran.
                check(std::isnan(y_),
                      "a regime that did NOT fire has NO prediction (NaN, not 0.0)");
                continue;
            }
            check(!std::isnan(y_), "a FIRING regime produced a prediction");
        }
        check(std::isnan(core->regimePrediction(kNStack)),
              "an out-of-range regime index reads NaN, never a neighbour's y_pred");
        check(std::isnan(core->regimePrediction(-1)), "a negative index reads NaN");

        if (weights_override.empty()) checkClosedForm(*core, kNStack);

        // ---- PHASE 5: the DECISION, end to end through the real ABI -----
        //
        // This is the ONE place the whole chain is graded together: the gate
        // struct crossed createCore, the core copied it into its internal form,
        // the models produced numbers on a live-built panel, and a side came
        // out. tests/decision_parity.py grades the RULE against the reference;
        // this grades the WIRING, which a rule-only test would pass whether or
        // not the core ever consulted the gate it was handed.
        const Decision dec_ = core->decide();
        checkEq<int64_t>(dec_.bar_ts_ms, d.panel_bar_ts_ms,
                         "the Decision names the bar the panel describes");

        // Recompute the vote from the per-regime flags the ABI exposes. Not a
        // second implementation of the rule -- regimeTriggeredLatest() IS the
        // core's own answer -- but it proves n_triggered, votes_long/short and
        // side all describe the SAME set of regimes.
        int votes_l_ = 0, votes_s_ = 0, gate_only_ = 0;
        for (int i = 0; i < kNStack; ++i) {
            const bool held_ = core->regimeFiredLatest(i);
            const bool voted_ = core->regimeTriggeredLatest(i);
            check(!(voted_ && !held_),
                  "no regime votes without its GATE having held first");
            if (held_ && !voted_) ++gate_only_;
            if (!voted_) continue;
            if (kStack[i].position > 0) ++votes_l_; else ++votes_s_;
        }
        check(!core->regimeTriggeredLatest(kNStack),
              "an out-of-range regime never reads as a vote");
        check(!core->regimeTriggeredLatest(-1), "nor does a negative index");
        checkEq<int>(dec_.n_triggered, votes_l_ + votes_s_,
                     "n_triggered is the VOTE count (Phase 5 redefined it): the "
                     "regimes that cleared their leg's threshold, not merely the "
                     "ones the gate let through");
        checkEq<int64_t>(d.votes_long, static_cast<int64_t>(votes_l_),
                         "votes_long matches the per-regime flags");
        checkEq<int64_t>(d.votes_short, static_cast<int64_t>(votes_s_),
                         "votes_short matches the per-regime flags");
        const int net_ = votes_l_ - votes_s_;
        const int want_side_ = (net_ * kGate.reverse > 0) ? +1
                             : ((net_ * kGate.reverse < 0) ? -1 : 0);
        checkEq<int>(dec_.side, want_side_,
                     "side == sign(net_count * reverse) -- trading.py:861 then "
                     "orb_bridge.py:168");
        check(dec_.fired == (want_side_ != 0), "fired == (side != 0)");
        checkEq<int64_t>(d.decisions_evaluated, 1,
                         "exactly ONE decision has been evaluated so far");

        if (dec_.n_triggered > 0) {
            const int win_ = core->winningRegimeIndex();
            check(win_ >= 0 && win_ < kNStack, "the representative regime is in the stack");
            if (win_ >= 0 && win_ < kNStack) {
                check(core->regimeTriggeredLatest(win_),
                      "and it is one that VOTED (not merely one whose gate held)");
                checkEq<double>(dec_.y_pred, core->regimePrediction(win_),
                                "Decision::y_pred is that regime's prediction");
                const bool long_ = kStack[win_].position > 0;
                checkEq<double>(dec_.threshold,
                                long_ ? kGate.threshold_long : kGate.threshold_short,
                                "the reported width is that regime's LEG's");
                checkEq<double>(dec_.threshold_center,
                                long_ ? kGate.threshold_center_long
                                      : kGate.threshold_center_short,
                                "the reported centre is that regime's LEG's");
                // The reported triple must EXPLAIN ITSELF: the y_pred in the log
                // line really does clear the centre +/- width printed beside it.
                check(long_ ? (dec_.y_pred > dec_.threshold_center + dec_.threshold)
                            : (dec_.y_pred < dec_.threshold_center - dec_.threshold),
                      "the reported y_pred clears the reported centre +/- width");
                if (net_ != 0) {
                    check((kStack[win_].position > 0) == (net_ > 0),
                          "and it sits on the MAJORITY leg, not the leg the "
                          "decision went against");
                }
            }
        } else {
            checkEq<int>(core->winningRegimeIndex(), -1,
                         "nothing voted, so no regime is representative");
            check(!dec_.fired, "and nothing fired");
        }

        // KLINE-BUILT -> SIGNAL-GENERATED. A real span now: the bar carries its
        // own emit stamp and decide() stamps the signal, both off system_clock.
        // Read from a FRESH diagnostics(): the span is computed INSIDE decide(),
        // so the snapshot taken before it cannot carry it — and a check against
        // the stale copy would have read 0 forever and called it a fast run.
        const CoreDiagnostics after_ = core->diagnostics();
        check(after_.bar_to_signal_us > 0,
              "the kline-built -> signal-generated span was MEASURED (>0us)");
        check(after_.bar_to_signal_us < 5'000'000,
              "and it is a latency, not a clock mismatch (<5s)");

        // IDEMPOTENT. A caller that logs and then re-reads must not manufacture
        // a second signal or a shorter latency.
        const Decision again_ = core->decide();
        checkEq<int>(again_.side, dec_.side, "decide() twice on one bar: same side");
        checkEq<int>(again_.n_triggered, dec_.n_triggered, "same vote count");
        check(again_.signal_emit_ns == dec_.signal_emit_ns,
              "and the SAME emit stamp -- not a fresh one");
        checkEq<int64_t>(core->diagnostics().decisions_evaluated, 1,
                         "and the bar is still counted ONCE");

        std::printf("     %d/%d regime(s) fired, %" PRId64 " prediction(s), "
                    "votes %dL/%dS (%d held but did not vote), side=%d "
                    "y_pred=%.9g predict_us=%" PRId64 " bar_to_signal_us=%" PRId64 "\n",
                    firing_, kNStack, d.predictions_computed, votes_l_, votes_s_,
                    gate_only_, dec_.side, dec_.y_pred, d.predict_us,
                    after_.bar_to_signal_us);
    }

    // ----------------------------------------------------------------- (4)
    // A BURST. Jump four buckets: the real bar closes and three flat bars are
    // drained behind it (rule 4). All four are popped from one tick, and only
    // the NEWEST may carry a panel.
    std::printf("\n[4] a 4-bar burst -> 3 stale bars SKIPPED, only the newest gets a panel\n");
    const int64_t before_computed_ = d.panels_computed;
    tick(*core, b699 + 7 * kPeriodMs + 10000, px * 1.004, 1.0, tid);
    got = drain(*core);
    checkEq<size_t>(got.size(), 4u, "one real bar plus three flat bars");
    d = core->diagnostics();
    checkEq<int64_t>(d.panels_computed, before_computed_ + 1,
                     "exactly ONE panel for the whole burst");
    checkEq<int64_t>(d.panels_skipped_stale_bar, 3,
                     "the three superseded bars are skipped and COUNTED");
    checkEq<int64_t>(d.panel_bar_ts_ms, got.back().bucket_open_ms,
                     "the panel belongs to the NEWEST bar of the burst");
    checkEq<int64_t>(d.panel_rows, static_cast<int64_t>(PANEL_BARS), "still 699 rows");
    checkEq<int64_t>(d.panel_cols, 65, "still 65 columns");

    // ----------------------------------------------------------------- (5)
    // The measured cost, on this path, for real. Phase 1 could only quote a
    // TA-Lib microbenchmark plus an estimate.
    std::printf("\n[5] measured per-bar panel cost on the LIVE path\n");
    std::vector<int64_t> us;
    int graded_ = 0;
    const int n_bench = bench_n > 0 ? bench_n : 20;
    int64_t ts = b699 + 8 * kPeriodMs + 10000;
    for (int i = 0; i < n_bench; ++i) {
        ts += kPeriodMs;
        px *= 1.0 + (w.next() - 0.5) * 0.004;
        tick(*core, ts, px, 1.0 + w.next(), tid);
        // A second trade in the same bucket so the bar is not a single fill.
        tick(*core, ts + 1000, px * 1.0001, 0.5, tid);
        got = drain(*core);
        const CoreDiagnostics dd = core->diagnostics();
        if (!got.empty() && dd.panel_rows == static_cast<int64_t>(PANEL_BARS)) {
            us.push_back(dd.feature_compute_us);
            // PHASE 4, ON EVERY BAR OF THE BENCH RUN. The first warm bar may
            // gate everything out (it does, on this walk), so grading the model
            // only there would leave the closed form UNCHECKED while the run
            // still printed a pass. Graded wherever a regime actually fires.
            if (weights_override.empty()) graded_ += checkClosedForm(*core, kNStack);
        }
    }
    d = core->diagnostics();
    if (weights_override.empty()) {
        // A gate that never graded anything is not a gate. The synthetic walk
        // is fixed and deterministic, so this is a property of the run, not of
        // the market.
        check(graded_ > 0,
              "the closed-form prediction check ran on at least one firing "
              "regime -- otherwise Phase 4's arithmetic is unverified here");
        checkEq<int64_t>(static_cast<int64_t>(graded_), d.predictions_computed,
                         "EVERY prediction the core computed was graded against "
                         "the closed form");
        std::printf("     closed-form checked on %d (regime, bar) prediction(s)\n",
                    graded_);
    }
    check(us.size() >= static_cast<size_t>(n_bench) - 1,
          "a panel on essentially every bar of the bench run");
    if (!us.empty()) {
        std::printf("     n=%zu  min=%.0f  p50=%.0f  p95=%.0f  max=%.0f us"
                    "   (bar period = %d s)\n",
                    us.size(), pct(us, 0.0), pct(us, 0.50), pct(us, 0.95), pct(us, 1.0),
                    kBarSec);
        std::printf("     run mean=%.0f us over %" PRId64 " panels, worst %" PRId64 " us\n",
                    d.panels_computed ? static_cast<double>(d.feature_compute_us_total)
                                        / static_cast<double>(d.panels_computed) : 0.0,
                    d.panels_computed, d.feature_compute_us_max);
        // A budget check, not a benchmark: the panel must finish inside the bar
        // it describes by an enormous margin, or the strategy is late by
        // construction. 1% of the bar period is already 9 s at 15m.
        check(pct(us, 1.0) < kBarSec * 10000.0,
              "worst panel is under 1% of the bar period");
    }

    checkEq<int64_t>(d.panel_errors, 0, "no panel error over the whole run");

    // ----------------------------------------------------------------- (5b)
    // STAGE 3, over the whole run. Two properties, and the second is the one
    // that stops "nothing fired" from reading as a pass:
    //   * the three r07x-gated regimes fired on ZERO bars, ever;
    //   * at least one FIRABLE regime fired on at least one bar, so the gate is
    //     demonstrably capable of returning true on this path.
    std::printf("\n[5b] the gate over the whole run\n");
    checkEq<int64_t>(d.regime_errors, 0, "no gate error over the whole run");
    checkEq<int64_t>(d.regime_evals, d.panels_computed,
                     "the gate ran on EVERY panel — one evaluation per panel");
    int64_t inert_fires_ = 0;
    int64_t firable_fires_ = 0;
    for (int i = 0; i < kNStack; ++i) {
        const int64_t n = core->regimeFireCount(i);
        std::printf("     regime[%d] %s fired %" PRId64 "/%" PRId64 " bar(s)\n",
                    i, i < kNFirable ? "firable" : "r07x   ", n, d.panels_computed);
        if (i < kNFirable) firable_fires_ += n; else inert_fires_ += n;
    }
    checkEq<int64_t>(inert_fires_, 0,
                     "the r07x-gated regimes fired on ZERO bars of the whole run");
    check(firable_fires_ > 0,
          "at least one firable regime DID fire — otherwise an all-False gate "
          "would satisfy every assertion above");
    check(d.regime_gate_us_max >= 0 && d.regime_gate_us_max < kBarSec * 1000000LL,
          "the gate's worst evaluation is inside the bar period");
    std::printf("     gate cost: last=%" PRId64 " us  worst=%" PRId64 " us over %"
                PRId64 " evaluation(s)\n",
                d.regime_gate_us, d.regime_gate_us_max, d.regime_evals);

    // ----------------------------------------------------------------- (5c)
    // RULE 7, the DROPPED-TRADE detector, on the LIVE path. The builder's own
    // self-tests pin the counting; this pins that the counters actually reach
    // the ABI, which is a different thing and is where a wiring omission would
    // hide.
    std::printf("\n[5c] the dropped-trade detector reaches the ABI\n");
    checkEq<int64_t>(d.trade_id_gaps, 0, "the synthetic feed's ids are contiguous");
    checkEq<int64_t>(d.trades_missing, 0, "so nothing is reported missing");
    {
        // Now SKIP 500 ids and require the next bar to say so.
        const int64_t gaps_before_ = d.trade_id_gaps;
        tid += 500;                       // 500 ids that never arrive
        ts += kPeriodMs;
        tick(*core, ts, px, 1.0, tid);
        ts += kPeriodMs;
        tick(*core, ts, px, 1.0, tid);    // closes the bar that carries the hole
        const std::vector<KlineBar> holed_ = drain(*core);
        const CoreDiagnostics dh = core->diagnostics();
        checkEq<int64_t>(dh.trade_id_gaps, gaps_before_ + 1,
                         "one gap EVENT crossed the ABI");
        checkEq<int64_t>(dh.trades_missing, 500,
                         "exactly 500 missing ids crossed the ABI");
        int64_t on_bars_ = 0;
        for (const KlineBar& b : holed_) on_bars_ += b.n_trades_missing;
        checkEq<int64_t>(on_bars_, 500,
                         "and the SAME 500 are attributed to a bar, not only to "
                         "the run total");
        d = dh;
    }

    // ----------------------------------------------------------------- (5d)
    // PHASE 5 over the whole run. A decision is taken on EVERY panel — in
    // runModels, alongside the predictions it is about — so the two counts must
    // agree exactly. They would not if the decision were taken lazily inside
    // decide(), which is the shape under which a bar nobody asked about would
    // silently have no decision at all.
    std::printf("\n[5d] the decision over the whole run\n");
    {
        const Decision last_ = core->decide();
        const CoreDiagnostics dd = core->diagnostics();
        checkEq<int64_t>(dd.decisions_evaluated, dd.panels_computed,
                         "one decision per PANEL -- taken with the predictions, "
                         "not lazily when someone asks");
        checkEq<int64_t>(dd.decisions_fired,
                         dd.decisions_long + dd.decisions_short,
                         "every fired decision took a side");
        checkEq<int64_t>(dd.decision_errors, 0, "no decision threw");
        if (dd.decision_errors > 0) {
            std::printf("     last decision error: %s\n",
                        core->lastDecisionError().c_str());
        }
        check(dd.bar_to_signal_us_max > 0,
              "the worst kline->signal span of the run was measured");
        checkEq<int64_t>(last_.bar_ts_ms, dd.panel_bar_ts_ms,
                         "the final decision names the final panel's bar");
        std::printf("     decisions=%" PRId64 " fired=%" PRId64 " (long=%" PRId64
                    " short=%" PRId64 ") errors=%" PRId64 "\n",
                    dd.decisions_evaluated, dd.decisions_fired, dd.decisions_long,
                    dd.decisions_short, dd.decision_errors);
        std::printf("     bar_to_signal_us last=%" PRId64 " worst=%" PRId64
                    "  (feature panel + regime gate + models + vote)\n",
                    dd.bar_to_signal_us, dd.bar_to_signal_us_max);
        std::printf("     NOTE: this driver's weights are a synthetic FIXTURE; how "
                    "often they clear the deployed gate says nothing about live. "
                    "The RULE is graded in tests/decision_parity.py.\n");
    }

    // ---- [5e] THE WALL-CLOCK FLUSH -------------------------------------
    // Graded here because live traffic CANNOT grade it: with an active symbol
    // a tick always rolls the bucket within milliseconds, so the flush is a
    // safety net that never trips and "it works" would rest on reasoning
    // alone. Measured on hydra 2026-08-20: 35 of 35 live bars were closed by a
    // tick and ZERO by the flush. The path that only runs when a symbol goes
    // quiet is exactly the path no live run exercises.
    std::printf("\n[5e] the wall-clock flush\n");
    {
        ts += kPeriodMs;
        tick(*core, ts, px, 1.0, ++tid);      // opens a fresh bucket
        drain(*core);                          // take whatever that rolled
        const int64_t open_ms_ = (ts / kPeriodMs) * kPeriodMs;
        const int64_t end_ms_  = open_ms_ + kPeriodMs;

        // NEGATIVE CONTROL, and the one that matters: a bucket whose end has
        // NOT passed must not be closed. Without this the test would pass just
        // as well if flushDue closed everything unconditionally.
        checkEq<int>(core->flushDueBuckets(end_ms_ - 1), 0,
                     "flush does NOT close a bucket that is still open");
        KlineBar probe_{};
        check(!core->barReady(&probe_),
              "and emits no bar while the bucket is still open");

        // Due now: the end has passed.
        checkEq<int>(core->flushDueBuckets(end_ms_), 1,
                     "flush CLOSES the bucket once its end has passed");
        KlineBar flushed_{};
        check(core->barReady(&flushed_), "the flushed bar reaches barReady()");
        checkEq<int64_t>(flushed_.bucket_open_ms, open_ms_,
                         "and it is the bucket that just ended");
        // close_trigger_recv_ns is 0 by contract: no tick triggered this, so
        // there is no recv->bar span and reporting one would invent a number.
        checkEq<uint64_t>(flushed_.close_trigger_recv_ns, 0ULL,
                          "a flushed bar reports NO recv->bar span");

        // IDEMPOTENT: polling again closes nothing and cannot double-emit.
        checkEq<int>(core->flushDueBuckets(end_ms_ + kPeriodMs), 0,
                     "flushing again closes nothing -- no open bucket remains");
        KlineBar again_{};
        check(!core->barReady(&again_), "and produces no second copy of it");
    }

    std::printf("\n[6] final diagnostics\n");
    std::printf("     bars_seen=%" PRId64 " contiguous=%" PRId64 "/%d panels=%" PRId64
                " skipped_not_warm=%" PRId64 " skipped_stale=%" PRId64 " errors=%" PRId64 "\n",
                d.bars_seen, d.contiguous_bars, core->warmupRequirement(), d.panels_computed,
                d.panels_skipped_not_warm, d.panels_skipped_stale_bar, d.panel_errors);
    std::printf("     models=%" PRId64 " feature_counts=%" PRId64 " (%" PRId64 ".."
                "%" PRId64 ") unit_scale=%" PRId64 " preds=%" PRId64 " nan_filled=%"
                PRId64 " nonfinite=%" PRId64 " model_errors=%" PRId64 "\n",
                d.models_loaded, d.model_feature_count_variants, d.model_features_min,
                d.model_features_max, d.model_unit_scale_features,
                d.predictions_computed, d.nan_features_filled,
                d.nonfinite_predictions, d.model_errors);
    if (d.model_errors > 0) {
        std::printf("     last model error: %s\n", core->lastModelError().c_str());
    }
    if (d.panel_errors > 0) {
        std::printf("     last panel error: %s\n", core->lastPanelError().c_str());
    }

    std::printf("\n=== %s: %d checks, %d failures ===\n",
                g_failures == 0 ? "INTEGRATION PASS" : "INTEGRATION FAIL",
                g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}

} // namespace

int main(int argc, char** argv)
{
    int bench_n = 0;
    std::string weights, tmp = "/tmp";
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--bench") == 0 && i + 1 < argc) bench_n = std::atoi(argv[++i]);
        // A real export, for a smoke test. The closed-form prediction checks
        // are skipped under it -- they only hold for the fixture, and pretending
        // otherwise would mean asserting numbers this file cannot derive.
        else if (std::strcmp(argv[i], "--weights") == 0 && i + 1 < argc) weights = argv[++i];
        else if (std::strcmp(argv[i], "--tmp") == 0 && i + 1 < argc) tmp = argv[++i];
    }
    // Reported, not left to terminate(): a width or column rejection must
    // arrive as a MESSAGE. An abort with empty output reads like "no bars".
    try {
        return run(bench_n, weights, tmp);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "core_integration_driver: %s\n", e.what());
        return 2;
    }
}
