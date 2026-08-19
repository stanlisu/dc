#include "kline_builder.hpp"

#include <algorithm>
#include <ctime>

namespace agamotto {
namespace {

// aef UpdateKind values we care about. Duplicated as constants rather than
// including the SDK header: the core must build standalone for the parity
// drivers, which have no SDK.
constexpr int kTradeUpdate = 6;
constexpr int kAggTradeUpdate = 7;

// Plausible epoch-millisecond range: 2001-09-09 .. 2286-11-20. This is not
// defensive boilerplate — it is load-bearing. The SHM Quote's own
// message.last_trade_ts reads 18359008543379257232 on the live feed (measured
// 2026-08-18), i.e. uninitialised garbage. Bucketing that value put every
// trade in one impossible bucket that never closed, and the run produced zero
// bars with no error anywhere. A range check turns that class of bug into a
// counted, visible reject instead of silence.
constexpr int64_t kMinPlausibleMs = 1'000'000'000'000LL;
constexpr int64_t kMaxPlausibleMs = 10'000'000'000'000LL;

uint64_t nowRealtimeNs()
{
	struct timespec ts {};
	clock_gettime(CLOCK_REALTIME, &ts);
	return static_cast<uint64_t>(ts.tv_sec) * 1'000'000'000ULL
	     + static_cast<uint64_t>(ts.tv_nsec);
}

} // namespace

KlineBuilder::KlineBuilder(int period_sec, int max_history)
  : mPeriodMs(static_cast<int64_t>(period_sec) * 1000),
    mMaxHistory(max_history > 0 ? static_cast<size_t>(max_history) : 1)
{
}

bool KlineBuilder::trySplice()
{
	// The quarantine is reunited with the live run ONLY when the two are
	// exactly adjacent. Anything else would be stitching across a hole, which
	// is the failure this whole mechanism exists to prevent.
	if (mPending.empty() || mHistory.empty()) {
		return false;
	}
	if (mPending.back().bucket_open_ms + mPeriodMs != mHistory.front().bucket_open_ms) {
		return false;
	}
	for (const KlineBar& h : mHistory) {
		mPending.push_back(h);
	}
	mHistory.swap(mPending);
	mPending.clear();
	++mSeamRepairs;
	return true;
}

void KlineBuilder::pushHistory(const KlineBar& b)
{
	// Contiguity is an INVARIANT of mHistory, not an aspiration: a 700-bar
	// window straddling a hole produces numbers that look fine and are not.
	// But the pre-gap run is not WRONG, only DISCONNECTED — so it is
	// quarantined, never destroyed, and can be reunited once the missing
	// buckets are ingested. Rule 6: the boot seam guarantees exactly one such
	// hole on every start, and clearing here discarded the whole backfill on
	// every start.
	if (!mHistory.empty()) {
		const int64_t expected = mHistory.back().bucket_open_ms + mPeriodMs;
		if (b.bucket_open_ms < expected) {
			// A REWIND, not a gap: this bucket is already covered by, or older
			// than, the retained run. Writing it would rewrite the past, and
			// calling it a discontinuity would report a BACKWARDS missing
			// range. Counted so a stale feed or a wrong-clock CSV is visible.
			++mRewoundBarsDropped;
			return;
		}
		if (b.bucket_open_ms > expected) {
			++mSeamGaps;
			mLastGapFromMs = expected;
			mLastGapToMs = b.bucket_open_ms - mPeriodMs;
			if (!mPending.empty()) {
				// A second hole opened while the first was still outstanding.
				// The older segment is now TWO holes from the live run, so
				// repairing it would need both closed; only the newer,
				// adjacent segment is kept. Counted, never silent.
				++mQuarantinesDiscarded;
			}
			mPending.swap(mHistory);
			mHistory.clear();
		}
	}
	mHistory.push_back(b);
	while (mHistory.size() > mMaxHistory) {
		mHistory.pop_front();
	}
	// The quarantine is capped too, or a long unrepaired run would grow it
	// without bound. Trimming from the front keeps the buckets NEAREST the
	// hole, which are the ones a splice needs.
	while (mPending.size() > mMaxHistory) {
		mPending.pop_front();
	}
}

int64_t KlineBuilder::bucketOf(int64_t ts_ms) const
{
	// Floor division. Binance buckets tile the UTC day from epoch, so plain
	// floor against the epoch is exactly their grid for any period dividing a
	// day. Guard the negative case even though exchange stamps are positive —
	// C++ integer division truncates toward zero, which would put a negative
	// stamp in the bucket ABOVE the one it belongs to.
	if (ts_ms >= 0) {
		return (ts_ms / mPeriodMs) * mPeriodMs;
	}
	return ((ts_ms - mPeriodMs + 1) / mPeriodMs) * mPeriodMs;
}

void KlineBuilder::startBucket(int64_t bucket_open_ms)
{
	// The bar is stamped from the tick that CLOSES it, which arrives later and
	// is passed to emitCurrent(). Holding a per-bucket recv time here as well
	// implied the bar was timed from its last contained trade — a different
	// quantity — so it is gone rather than left to mislead.
	mCur = Accum{};
	mCur.bucket_open_ms = bucket_open_ms;
	mHaveOpenBucket = true;
}

void KlineBuilder::emitFlat(int64_t bucket_open_ms)
{
	// Binance still publishes a kline for a bucket with no trades: OHLC all
	// equal to the previous close, zero volume. Dropping these would compress
	// the time axis and shift every rolling window downstream.
	KlineBar b{};
	b.bucket_open_ms = bucket_open_ms;
	b.bucket_close_ms = bucket_open_ms + mPeriodMs - 1;
	b.open = b.high = b.low = b.close = mPrevClose;
	b.volume = 0.0;
	b.quote_volume = 0.0;
	b.number_of_trades = 0;
	b.taker_buy_base_volume = 0.0;
	b.taker_buy_quote_volume = 0.0;
	b.aggressor_source = KlineBar::AggressorSource::NONE;
	b.n_trades_unclassified = 0;
	b.from_backfill = false;
	b.close_trigger_recv_ns = 0;   // no tick closed it; excluded from latency
	b.bar_emit_ns = nowRealtimeNs();
	mReady.push_back(b);
	pushHistory(b);
	++mBarsSeen;
}

void KlineBuilder::emitCurrent(uint64_t close_trigger_recv_ns)
{
	if (!mHaveOpenBucket) {
		return;
	}
	if (mCurIsPartial) {
		// Rule 3: we attached mid-bucket, so this bar is missing the trades
		// that happened before we attached. Emitting it would put a real-
		// looking bar with implausibly low volume into the series.
		++mPartialBucketsDropped;
		mCurIsPartial = false;
		mHaveOpenBucket = false;
		if (mCur.has_any_trade) {
			mPrevClose = mCur.close;
			mHaveClose = true;
		}
		return;
	}

	KlineBar b{};
	b.bucket_open_ms = mCur.bucket_open_ms;
	b.bucket_close_ms = mCur.bucket_open_ms + mPeriodMs - 1;
	if (mCur.has_any_trade) {
		b.open = mCur.open;
		b.high = mCur.high;
		b.low = mCur.low;
		b.close = mCur.close;
	} else {
		b.open = b.high = b.low = b.close = mPrevClose;
	}
	b.volume = mCur.volume;
	b.quote_volume = mCur.quote_volume;
	b.number_of_trades = mCur.n_trades;
	b.taker_buy_base_volume = mCur.taker_buy_base;
	b.taker_buy_quote_volume = mCur.taker_buy_quote;
	b.n_trades_unclassified = mCur.n_unclassified;

	// Provenance is per BAR: if any trade in it needed the quote rule, the
	// taker_buy_* columns are approximations and must not be reported as exact.
	if (mCur.any_quote_rule) {
		b.aggressor_source = KlineBar::AggressorSource::QUOTE_RULE;
	} else if (mCur.any_exact) {
		b.aggressor_source = KlineBar::AggressorSource::EXACT_MAKER_FLAG;
	} else {
		b.aggressor_source = KlineBar::AggressorSource::NONE;
	}

	b.from_backfill = false;
	b.close_trigger_recv_ns = close_trigger_recv_ns;
	b.bar_emit_ns = nowRealtimeNs();

	mPrevClose = b.close;
	mHaveClose = true;
	mReady.push_back(b);
	pushHistory(b);
	++mBarsSeen;
	mHaveOpenBucket = false;
}

void KlineBuilder::applyTrade(const TickEvent& ev)
{
	if (!mCur.has_any_trade) {
		mCur.open = ev.last_px;
		mCur.high = ev.last_px;
		mCur.low = ev.last_px;
		mCur.has_any_trade = true;
	} else {
		mCur.high = std::max(mCur.high, ev.last_px);
		mCur.low = std::min(mCur.low, ev.last_px);
	}
	mCur.close = ev.last_px;

	const double quote_ = ev.last_px * ev.last_qty;
	mCur.volume += ev.last_qty;
	mCur.quote_volume += quote_;
	mCur.n_trades += 1;

	// Aggressor: exact when the feed populated the maker flag, else the quote
	// rule (at/above the ask = buy-aggressor, at/below the bid = sell). A trade
	// strictly inside the spread cannot be sided by either and is COUNTED, not
	// guessed — silently calling it a buy would bias taker_buy_* one way.
	int side_ = ev.aggressor_is_buy;
	if (side_ >= 0) {
		mCur.any_exact = true;
	} else if (mLastAsk > 0.0 && ev.last_px >= mLastAsk) {
		side_ = 1;
		mCur.any_quote_rule = true;
	} else if (mLastBid > 0.0 && ev.last_px <= mLastBid) {
		side_ = 0;
		mCur.any_quote_rule = true;
	} else {
		++mCur.n_unclassified;
	}

	if (side_ == 1) {
		mCur.taker_buy_base += ev.last_qty;
		mCur.taker_buy_quote += quote_;
	}
}

void KlineBuilder::onTick(const TickEvent& ev)
{
	if (ev.has_book) {
		if (ev.bid_px > 0.0) mLastBid = ev.bid_px;
		if (ev.ask_px > 0.0) mLastAsk = ev.ask_px;
	}

	if (ev.update_kind == kAggTradeUpdate) {
		// Rule 1: counted, never applied. These are the same fills as kind 6,
		// re-aggregated; applying both doubles volume and number_of_trades.
		++mAggTradeUpdates;
		return;
	}
	if (ev.update_kind != kTradeUpdate || !ev.has_trade) {
		return;
	}
	++mTradeUpdates;

	if (ev.last_qty <= 0.0 || ev.last_px <= 0.0) {
		return;
	}

	// Rule 2: drop replays / repeated slots. Ids are monotonic per symbol
	// within the trade stream.
	if (mHaveLastTradeId && ev.last_trade_id != 0 && ev.last_trade_id <= mLastTradeId) {
		++mDuplicatesDropped;
		return;
	}
	if (ev.last_trade_id != 0) {
		mLastTradeId = ev.last_trade_id;
		mHaveLastTradeId = true;
	}

	// Bucketing uses the exchange's event time for the trade — Binance's own
	// kline assignment rule. Rejected loudly (counted) when implausible: see
	// kMinPlausibleMs.
	const int64_t ts_ms_ = static_cast<int64_t>(ev.last_trade_ts_ms);
	if (ts_ms_ < kMinPlausibleMs || ts_ms_ >= kMaxPlausibleMs) {
		++mBadTimestampDropped;
		return;
	}
	const int64_t bucket_ = bucketOf(ts_ms_);

	if (!mHaveOpenBucket) {
		startBucket(bucket_);
		mCurIsPartial = true;   // Rule 3: attached mid-bucket
		applyTrade(ev);
		return;
	}

	if (bucket_ < mCur.bucket_open_ms) {
		// Rule 5, mirrored: a trade stamped in an ALREADY-CLOSED bucket cannot
		// be applied without rewriting history. Counted so it is visible.
		++mLateTradesDropped;
		return;
	}

	if (bucket_ == mCur.bucket_open_ms) {
		applyTrade(ev);
		return;
	}

	// New bucket: close the open one, then walk any wholly empty buckets one
	// at a time so each gets its own flat bar (rule 4).
	emitCurrent(ev.recv_ts_ns);
	for (int64_t b = mCur.bucket_open_ms + mPeriodMs; b < bucket_; b += mPeriodMs) {
		if (mHaveClose) {
			emitFlat(b);
		}
	}
	startBucket(bucket_);
	applyTrade(ev);
}

bool KlineBuilder::pop(KlineBar* out)
{
	if (mReady.empty()) {
		return false;
	}
	*out = mReady.front();
	mReady.pop_front();
	return true;
}

bool KlineBuilder::ingestBackfill(const KlineBar* bars, int n)
{
	if (bars == nullptr || n <= 0) {
		return false;
	}
	// Validate the WHOLE run before mutating anything: a half-ingested backfill
	// that then reports failure leaves the core in a state nobody can reason
	// about.
	for (int i = 0; i < n; ++i) {
		if (bars[i].bucket_open_ms % mPeriodMs != 0) {
			return false;   // off-grid: cannot be a Binance bucket
		}
		if (i > 0 && bars[i].bucket_open_ms != bars[i - 1].bucket_open_ms + mPeriodMs) {
			return false;   // gap or out-of-order within the run
		}
	}

	// How the run joins what we already hold. Checked against the RETAINED
	// history, not against the open live bucket: the previous version compared
	// with mCur.bucket_open_ms, which is still 0 before the first tick, so a
	// second overlapping backfill sailed through and double-counted.
	if (mHistory.empty()) {
		for (int i = 0; i < n; ++i) {
			mHistory.push_back(bars[i]);
		}
	} else if (bars[n - 1].bucket_open_ms + mPeriodMs == mHistory.front().bucket_open_ms) {
		// PREPEND — this is the seam fix. The run ends exactly where the
		// retained history begins, so it closes the hole left by discarding the
		// partial bucket we attached on.
		for (int i = n - 1; i >= 0; --i) {
			mHistory.push_front(bars[i]);
		}
	} else if (bars[0].bucket_open_ms == mHistory.back().bucket_open_ms + mPeriodMs) {
		for (int i = 0; i < n; ++i) {
			mHistory.push_back(bars[i]);
		}
	} else {
		// Overlaps, or leaves a hole at either end. Refused outright rather
		// than stitched, since either would silently break contiguity.
		return false;
	}

	// Reunite the quarantined pre-gap run if this run just closed the hole.
	// MUST run BEFORE the ring trim: trimming first pops exactly the bars a
	// prepend just added when the ring is already full, which would leave the
	// splice permanently one bucket short with ingestBackfill still reporting
	// success.
	trySplice();

	while (mHistory.size() > mMaxHistory) {
		mHistory.pop_front();
	}

	// Only advance the live-build seed when the backfill is the NEWEST thing we
	// hold; a prepend describes older history and must not rewrite prev-close.
	if (!mHaveClose || bars[n - 1].bucket_open_ms >= mHistory.back().bucket_open_ms) {
		mPrevClose = mHistory.back().close;
		mHaveClose = true;
	}
	mBarsSeen += n;
	return true;
}

} // namespace agamotto
