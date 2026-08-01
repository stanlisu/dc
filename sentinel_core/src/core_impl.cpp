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
#include <array>
#include <cmath>
#include <limits>    // std::fabs — gcc 8 does not pull this in transitively
#include <deque>
#include <fstream>
#include "codes_generated.hpp"
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

        // Optional bar dump for reconciliation against the reference bot.
        // Written HERE, in the private core, because the bar schema is IP — the
        // public shell must not learn the column set. Headers are CODED for the
        // mapped columns; passthrough names (OHLCV, book, timestamps) stay real
        // because the obfuscation map does not cover them.
        try { mBarsCsv = p.str("bars_csv"); } catch (const std::exception&) { mBarsCsv.clear(); }
        try { mSymbol = p.str("symbol"); } catch (const std::exception&) { mSymbol = "UNKNOWN"; }

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
                dumpBar(closed);
                while (mBuf.size() > static_cast<size_t>(BUFFER_MAXLEN)) mBuf.pop_front();
                mPendingBarTsNs = closed.bucket_ms * 1000000LL;
                mHasPending = true;
            }
        }
    }

    void dumpBar(const Bar& b)
    {
        if (mBarsCsv.empty()) return;
        const bool need_header = !mBarsHeaderWritten;
        std::ofstream fh(mBarsCsv, std::ios::app);
        if (!fh) return;   // WHY: dump is diagnostic; a bad path must not kill the strategy
        if (need_header) {
            fh << "timestamp_ns,symbol,open,high,low,close,volume,buy_vol,sell_vol,n_trades,"
               << "vwap," << codes::F_TRADE_IMBALANCE << ",bid_price,bid_amount,ask_price,ask_amount";
            for (int i = 0; i < BOOK_LEVELS; ++i)
                fh << ",bids_" << i << "_price,bids_" << i << "_qty"
                   << ",asks_" << i << "_price,asks_" << i << "_qty";
            fh << ",depth_bid_L1,depth_bid_L3,depth_bid_L5,depth_ask_L1,depth_ask_L3,depth_ask_L5"
               << ",mark_price,index_price,funding_rate,predicted_funding_rate,open_interest"
               << "," << codes::F_LIQ_LONG_NOTIONAL << "," << codes::F_LIQ_SHORT_NOTIONAL
               << ",liq_long_count,liq_short_count,liq_total_count"
               << "," << codes::F_CYCLE_PROGRESS << "," << codes::F_SECS_TO_BOUNDARY << "\n";
            mBarsHeaderWritten = true;
        }
        fh.precision(17);
        fh << (b.bucket_ms * 1000000LL) << "," << mSymbol << ","
           << b.open << "," << b.high << "," << b.low << "," << b.close << ","
           << b.volume << "," << b.buy_vol << "," << b.sell_vol << "," << b.n_trades << ","
           << b.vwap << "," << b.trade_imbalance << ","
           << b.bid_price << "," << b.bid_amount << "," << b.ask_price << "," << b.ask_amount;
        for (int i = 0; i < BOOK_LEVELS; ++i)
            fh << "," << b.bids_price[i] << "," << b.bids_qty[i]
               << "," << b.asks_price[i] << "," << b.asks_qty[i];
        fh << "," << b.depth_bid_L1 << "," << b.depth_bid_L3 << "," << b.depth_bid_L5
           << "," << b.depth_ask_L1 << "," << b.depth_ask_L3 << "," << b.depth_ask_L5
           << "," << b.mark_price << "," << b.index_price << "," << b.funding_rate
           << "," << b.predicted_funding_rate << "," << b.open_interest
           << "," << b.liq_long_notional << "," << b.liq_short_notional
           << "," << b.liq_long_count << "," << b.liq_short_count
           << "," << b.liq_total_count << "," << b.cycle_progress
           << "," << b.secs_to_boundary << "\n";
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
    int  anchorFrameSize() const override { return static_cast<int>(mOwnFrame.size()); }
    const double* anchorFrame() const override
    {
        // Populated by decide() on the anchor. Empty until the first scored bar,
        // so a peer cannot consume a frame that was never computed.
        return mOwnFrame.empty() ? nullptr : mOwnFrame.data();
    }
    void setAnchorFrame(const double* frame, int n, int64_t barTsNs) override
    {
        if (frame == nullptr || n <= 0) {
            // Absent anchor state is NOT silently tolerated: the peer keeps its
            // history but records nothing for this bar, so the row aligns to NaN
            // and the model's missing-feature check fires rather than predicting
            // from a substituted value.
            return;
        }
        if (n != FeatureEngine::ANCHOR_COLS) {
            throw std::runtime_error(
                "anchor frame width " + std::to_string(n) + " != expected "
                + std::to_string(FeatureEngine::ANCHOR_COLS)
                + " — publisher/consumer version skew");
        }
        std::array<double, FeatureEngine::ANCHOR_COLS> row{};
        for (int i = 0; i < n; ++i) row[static_cast<size_t>(i)] = frame[i];
        mAnchorHist[barTsNs] = row;
        // Bound the history the same way the bar buffer is bounded.
        while (mAnchorHist.size() > static_cast<size_t>(BUFFER_MAXLEN) * 2)
            mAnchorHist.erase(mAnchorHist.begin());
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

        if (mIsAnchor) {
            // Publish OUR slim frame for the bar we are about to score, so peers
            // join on the same bar_ts rather than on whatever arrived last.
            mOwnFrame.assign(FeatureEngine::ANCHOR_COLS, 0.0);
            FeatureEngine::extractAnchorRow(names, cols, win.size() - 2, mOwnFrame.data());
            mOwnFrameBarTs = win[win.size() - 2].bucket_ms * 1000000LL;
        } else {
            // Peers: align the anchor history to OUR bars BY bar_ts. Rows with no
            // anchor bar stay NaN — never carried forward from a neighbour, which
            // would silently invent cross-features.
            std::vector<double> anchor(win.size() * FeatureEngine::ANCHOR_COLS,
                                       std::numeric_limits<double>::quiet_NaN());
            size_t matched = 0;
            for (size_t i = 0; i < win.size(); ++i) {
                auto it = mAnchorHist.find(win[i].bucket_ms * 1000000LL);
                if (it == mAnchorHist.end()) continue;
                for (int c = 0; c < FeatureEngine::ANCHOR_COLS; ++c)
                    anchor[i * FeatureEngine::ANCHOR_COLS + c] = it->second[static_cast<size_t>(c)];
                ++matched;
            }
            mLastAnchorMatched = matched;
            mFeatures->addAnchorCrossFeatures(names, cols, anchor);
        }

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

    std::map<int64_t, std::array<double, FeatureEngine::ANCHOR_COLS>> mAnchorHist;
    std::vector<double> mOwnFrame;      // anchor role: our slim frame to publish
    int64_t mOwnFrameBarTs{0};
    int64_t mAnchorBarTs{0};
    size_t  mLastAnchorMatched{0};

    Decision mHeld;
    int mHeldLeft{0};

    std::string mBarsCsv;
    std::string mSymbol{"UNKNOWN"};
    bool mBarsHeaderWritten{false};
};

} // namespace

std::unique_ptr<ICore> makeCore(const std::string& algo_params_json)
{
    return std::unique_ptr<ICore>(new Core(algo_params_json));
}

bool coreIsRealImplementation() { return true; }

const char* coreBuildTag() { return "core-" MJOLNIR_CORE_GITSHA; }

} // namespace mjolnir
