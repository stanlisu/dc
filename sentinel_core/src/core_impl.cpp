// Private core — implements the public opaque contract.
//
// Wires the four verified modules into the live path:
//   tick -> BarBuilder -> rolling buffer -> FeatureEngine -> RegimeGate ->
//   ModelRunner -> vote/threshold -> hold-TTL -> Decision
//
// The numerical chain is gated by tests/m1_gate.py (variant-A parity against
// production: 2085/2085 live firings, corr 1.000000). What lives HERE beyond
// that chain is the live plumbing — buffering, warmup, the vote, and hold-TTL.
//
// Design rules carried from the reference, each load-bearing:
//   * BUFFER_MAXLEN = 1000 bars (trading.py) — the feature window
//   * the scored row is iloc[-2], NOT the closing bar
//   * regime predicates are evaluated over the FULL panel (quantiles!)
//   * hold-TTL holds a fired signal through <= HOLD_TTL_BARS None cycles;
//     a fire re-arms it, a flip passes straight through, 0 = flat-on-None
//   * required config keys have NO defaults — a missing key raises
#include "mjolnir_core.hpp"

#include "bar_builder.hpp"
#include "feature_engine.hpp"
#include "regime_gate.hpp"
#include "model_runner.hpp"

#include <algorithm>
#include <deque>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef MJOLNIR_CORE_GITSHA
#define MJOLNIR_CORE_GITSHA "unknown"
#endif

namespace mjolnir {
namespace {

constexpr int BUFFER_MAXLEN = 1000;

std::vector<std::string> splitCsv(const std::string& s, char d = ',')
{
    std::vector<std::string> o;
    std::stringstream ss(s);
    std::string it;
    while (std::getline(ss, it, d)) o.push_back(it);
    return o;
}

std::string trimq(std::string s)
{
    while (!s.empty() && (s.back() == '\r' || s.back() == ' ' || s.back() == '"')) s.pop_back();
    size_t b = 0;
    while (b < s.size() && (s[b] == ' ' || s[b] == '"')) ++b;
    return s.substr(b);
}

// Minimal JSON scalar reader. algo_params.json is written by our own launcher,
// so a full parser is unnecessary — but a MISSING key must raise rather than
// default, which is why there is no get-with-fallback here.
class Params {
  public:
    explicit Params(const std::string& path)
    {
        std::ifstream fh(path);
        if (!fh) throw std::runtime_error("cannot open algo_params: " + path);
        std::stringstream ss;
        ss << fh.rdbuf();
        mRaw = ss.str();
    }

    std::string str(const std::string& key) const
    {
        const std::string v = rawValue(key);
        return trimq(v);
    }
    double num(const std::string& key) const
    {
        const std::string v = rawValue(key);
        try { return std::stod(v); }
        catch (...) { throw std::runtime_error("algo_params: '" + key + "' is not numeric"); }
    }
    int integer(const std::string& key) const { return static_cast<int>(num(key)); }
    bool boolean(const std::string& key) const
    {
        const std::string v = trimq(rawValue(key));
        if (v == "true" || v == "1") return true;
        if (v == "false" || v == "0") return false;
        throw std::runtime_error("algo_params: '" + key + "' is not boolean");
    }

  private:
    std::string rawValue(const std::string& key) const
    {
        const std::string needle = "\"" + key + "\"";
        size_t p = mRaw.find(needle);
        if (p == std::string::npos)
            throw std::runtime_error("algo_params: required key '" + key + "' is missing "
                                     "(no default — see CLAUDE.md on silent fallbacks)");
        p = mRaw.find(':', p + needle.size());
        if (p == std::string::npos) throw std::runtime_error("algo_params: malformed '" + key + "'");
        ++p;
        while (p < mRaw.size() && (mRaw[p] == ' ' || mRaw[p] == '\t')) ++p;
        size_t e = p;
        while (e < mRaw.size() && mRaw[e] != ',' && mRaw[e] != '\n' && mRaw[e] != '}') ++e;
        std::string v = mRaw.substr(p, e - p);
        while (!v.empty() && (v.back() == ' ' || v.back() == '\r')) v.pop_back();
        return v;
    }
    std::string mRaw;
};

struct StackEntry {
    std::string regime;      // as written in the stack (code OR real name)
    std::string position;    // "long" / "short"
    double threshold{0.0};
    std::string dir;         // weights subdirectory
};

// Stack CSV: regime,model,position,optimal_threshold,...
std::vector<StackEntry> loadStack(const std::string& path, const std::string& subsetFilter)
{
    std::ifstream fh(path);
    if (!fh) throw std::runtime_error("cannot open regime stack: " + path);
    std::string line;
    if (!std::getline(fh, line)) throw std::runtime_error("empty regime stack: " + path);
    const auto hdr = splitCsv(line);
    auto col = [&](const char* n) -> int {
        for (size_t i = 0; i < hdr.size(); ++i) if (trimq(hdr[i]) == n) return static_cast<int>(i);
        throw std::runtime_error(std::string("regime stack missing column: ") + n);
    };
    const int c_reg = col("regime"), c_pos = col("position"), c_thr = col("optimal_threshold");

    std::vector<StackEntry> out;
    while (std::getline(fh, line)) {
        if (line.empty()) continue;
        const auto f = splitCsv(line);
        if (static_cast<int>(f.size()) <= c_thr) continue;
        StackEntry e;
        e.dir = trimq(f[c_reg]);            // weights dir is the stack's regime string
        e.regime = e.dir;
        e.position = trimq(f[c_pos]);
        const std::string t = trimq(f[c_thr]);
        if (t.empty()) throw std::runtime_error("regime stack row has empty optimal_threshold");
        e.threshold = std::stod(t);
        // |threshold| >= 2bps is enforced offline when the stack is built; assert
        // it here too so a hand-edited stack cannot deploy an always-on signal.
        if (std::fabs(e.threshold) < 0.0002)
            throw std::runtime_error("regime stack threshold below the 2bps floor: " + t);
        if (!subsetFilter.empty() && e.dir.rfind(subsetFilter, 0) != 0) continue;
        out.push_back(e);
    }
    if (out.empty()) throw std::runtime_error("regime stack has no usable rows: " + path);
    return out;
}

class Core final : public ICore {
  public:
    explicit Core(const std::string& paramsPath)
    {
        Params p(paramsPath);
        mBarSec = p.integer("bar_sec");
        mTargetSec = p.integer("target_sec");
        mWarmupBars = p.integer("warmup_fire_gate_bars");
        mMinSignalCount = p.integer("min_signal_count");
        mReverse = p.integer("reverse");
        mHoldTtlBars = p.integer("hold_ttl_bars");
        mIsAnchor = p.boolean("is_anchor");
        const std::string weights = p.str("weights_dir");
        const std::string stackPath = p.str("regime_stack_csv");
        std::string subset;
        try { subset = p.str("regime_subset"); } catch (const std::exception&) { subset = ""; }

        mBuilder.reset(new BarBuilder(mBarSec, mTargetSec));
        mFeatures.reset(new FeatureEngine({30, 60, 300, 900}, mBarSec, mTargetSec));

        mStack = loadStack(stackPath, subset);
        for (const auto& e : mStack) {
            if (mModels.count(e.dir)) continue;
            auto mr = std::unique_ptr<ModelRunner>(new ModelRunner());
            mr->load(weights + "/" + e.dir);
            mModels.emplace(e.dir, std::move(mr));
        }
    }

    // ---- ingest ----------------------------------------------------------
    void onTick(const TickEvent& ev) override
    {
        if (ev.has_book) {
            mBuilder->onBookTicker(ev.bid_px[0], ev.bid_qty[0], ev.ask_px[0], ev.ask_qty[0],
                                   static_cast<int64_t>(ev.exchange_ts_ns / 1000000));
            mBuilder->onDepth(ev.bid_px, ev.bid_qty, ev.ask_px, ev.ask_qty, ev.n_levels,
                              static_cast<int64_t>(ev.exchange_ts_ns / 1000000));
        }
        if (ev.mark_px != 0.0 || ev.index_px != 0.0) {
            mBuilder->onMarkPrice(ev.mark_px, ev.index_px, ev.funding_rate, ev.funding_rate,
                                  static_cast<int64_t>(ev.exchange_ts_ns / 1000000));
        }
        if (ev.liq_is_long >= 0) {
            mBuilder->onLiquidation(ev.liq_is_long == 1, ev.liq_notional,
                                    static_cast<int64_t>(ev.exchange_ts_ns / 1000000));
        }
        if (ev.open_interest > 0.0) {
            mBuilder->setOpenInterest(ev.open_interest,
                                      static_cast<int64_t>(ev.exchange_ts_ns / 1000000));
        }
        if (ev.has_trade) {
            Bar closed;
            // aggressor_is_buy == 1 means the taker BOUGHT, i.e. the buyer was
            // NOT the maker — the builder's is_buyer_maker is the inverse.
            const bool is_buyer_maker = (ev.aggressor_is_buy == 0);
            if (mBuilder->onTrade(ev.last_px, ev.last_qty, is_buyer_maker,
                                  static_cast<int64_t>(ev.exchange_ts_ns / 1000000), 1, &closed)) {
                mBuf.push_back(closed);
                while (mBuf.size() > static_cast<size_t>(BUFFER_MAXLEN)) mBuf.pop_front();
                mPendingBarTsNs = closed.bucket_ms * 1000000LL;
                mHasPending = true;
            }
        }
    }

    bool barReady(int64_t* barTsNs) override
    {
        if (!mHasPending) return false;
        if (barTsNs) *barTsNs = mPendingBarTsNs;
        mHasPending = false;
        return true;
    }

    bool isWarm() const override { return static_cast<int>(mBuf.size()) >= mWarmupBars; }
    int  barsBuffered() const override { return static_cast<int>(mBuf.size()); }

    // ---- anchor bus -------------------------------------------------------
    bool isAnchor() const override { return mIsAnchor; }
    int  anchorFrameSize() const override { return static_cast<int>(mAnchorFrame.size()); }
    const double* anchorFrame() const override
    {
        return mAnchorFrame.empty() ? nullptr : mAnchorFrame.data();
    }
    void setAnchorFrame(const double* frame, int n, int64_t barTsNs) override
    {
        if (frame == nullptr || n <= 0) {
            mAnchorFrame.clear();
            mAnchorBarTs = 0;
            return;
        }
        mAnchorFrame.assign(frame, frame + n);
        mAnchorBarTs = barTsNs;
    }

    // ---- decide -----------------------------------------------------------
    Decision decide() override
    {
        Decision d;
        d.bar_ts_ns = mPendingBarTsNs;
        if (mBuf.size() < 2) return applyHoldTtl(d);

        const std::vector<Bar> win(mBuf.begin(), mBuf.end());
        std::vector<std::string> names;
        std::vector<std::vector<double>> cols;
        mFeatures->compute(win, names, cols);
        FeaturePanel panel(names, cols);

        // iloc[-2]: the row BEFORE the closing bar.
        const size_t row = win.size() - 2;

        int long_count = 0, short_count = 0;
        double long_thr = 0.0, short_thr = 0.0;
        double y_long = 0.0, y_short = 0.0;
        uint16_t win_long = 0, win_short = 0;
        bool have_long = false, have_short = false;

        for (const auto& e : mStack) {
            // Full-panel evaluation: quantile predicates are meaningless on one row.
            const auto mask = applyFilterMask(panel, e.regime, e.position);
            if (mask.size() <= row || !mask[row]) continue;

            const double y = mModels.at(e.dir)->predictRow(panel, row);
            const uint16_t code = regimeCodeOf(e.regime);
            if (e.position == "long" && y > e.threshold) {
                ++long_count;
                if (!have_long || e.threshold < long_thr) long_thr = e.threshold;
                y_long = y; win_long = code; have_long = true;
            } else if (e.position == "short" && y < e.threshold) {
                ++short_count;
                if (!have_short || e.threshold > short_thr) short_thr = e.threshold;
                y_short = y; win_short = code; have_short = true;
            }
        }

        if (long_count >= mMinSignalCount && long_count > short_count) {
            d.fired = true; d.side = 1 * mReverse;
            d.y_pred = y_long; d.threshold = long_thr;
            d.n_triggered = long_count; d.winning_regime_code = win_long;
        } else if (short_count >= mMinSignalCount && short_count > long_count) {
            d.fired = true; d.side = -1 * mReverse;
            d.y_pred = y_short; d.threshold = short_thr;
            d.n_triggered = short_count; d.winning_regime_code = win_short;
        }
        return applyHoldTtl(d);
    }

    int dumpPredictions(uint16_t* codes, int* positions, double* preds, int cap) const override
    {
        if (mBuf.size() < 2) return 0;
        const std::vector<Bar> win(mBuf.begin(), mBuf.end());
        std::vector<std::string> names;
        std::vector<std::vector<double>> cols;
        mFeatures->compute(win, names, cols);
        FeaturePanel panel(names, cols);
        const size_t row = win.size() - 2;
        int n = 0;
        for (const auto& e : mStack) {
            if (n >= cap) break;
            codes[n] = regimeCodeOf(e.regime);
            positions[n] = (e.position == "long") ? 1 : -1;
            preds[n] = mModels.at(e.dir)->predictRow(panel, row);
            ++n;
        }
        return n;
    }

  private:
    // Hold a fired signal through <= mHoldTtlBars consecutive non-firing bars.
    // A fire re-arms the TTL; a flip passes through immediately; 0 disables the
    // hold entirely (legacy flat-on-None).
    Decision applyHoldTtl(Decision d)
    {
        if (mHoldTtlBars <= 0) { mHeld = Decision{}; mHeldLeft = 0; return d; }

        if (d.fired) {
            mHeld = d;
            mHeldLeft = mHoldTtlBars;      // re-arm
            return d;
        }
        if (mHeldLeft > 0) {
            --mHeldLeft;
            Decision h = mHeld;
            h.bar_ts_ns = d.bar_ts_ns;     // held signal, current bar
            return h;
        }
        mHeld = Decision{};
        return d;
    }

    static uint16_t regimeCodeOf(const std::string& s)
    {
        // Stack rows may carry a code ("rNNN...") or a real name; report the
        // numeric code when it is a code, 0 otherwise (never a real name — the
        // public side must not see one).
        if (s.size() >= 4 && s[0] == 'r') {
            try { return static_cast<uint16_t>(std::stoi(s.substr(1, 3))); } catch (...) {}
        }
        return 0;
    }

    int mBarSec{0}, mTargetSec{0}, mWarmupBars{0};
    int mMinSignalCount{1}, mReverse{1}, mHoldTtlBars{0};
    bool mIsAnchor{false};

    std::unique_ptr<BarBuilder> mBuilder;
    std::unique_ptr<FeatureEngine> mFeatures;
    std::deque<Bar> mBuf;
    std::vector<StackEntry> mStack;
    std::map<std::string, std::unique_ptr<ModelRunner>> mModels;

    int64_t mPendingBarTsNs{0};
    bool mHasPending{false};

    std::vector<double> mAnchorFrame;
    int64_t mAnchorBarTs{0};

    Decision mHeld;
    int mHeldLeft{0};
};

} // namespace

std::unique_ptr<ICore> makeCore(const std::string& algo_params_json)
{
    return std::unique_ptr<ICore>(new Core(algo_params_json));
}

bool coreIsRealImplementation() { return true; }

const char* coreBuildTag() { return "core-" MJOLNIR_CORE_GITSHA; }

} // namespace mjolnir
