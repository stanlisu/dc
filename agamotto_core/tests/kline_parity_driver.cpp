// Standalone driver for the 15m kline builder. No SDK, no live feed.
//
//   --selftest                       synthetic scenarios, asserts, exit != 0 on failure
//   --ticks T.csv --klines K.csv     replay recorded ticks, diff vs Binance's own klines
//
// tick csv:  trade_ts_ms,trade_id,px,qty,bid,ask,kind      (kind 6 = trade, 7 = aggTrade)
// kline csv: open_ms,open,high,low,close,volume,quote_volume,n_trades,tb_base,tb_quote
#include "kline_builder.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using agamotto::KlineBar;
using agamotto::KlineBuilder;
using agamotto::TickEvent;

namespace {

int g_failures = 0;

void check(bool cond, const char* what)
{
	if (!cond) {
		std::printf("  FAIL: %s\n", what);
		++g_failures;
	} else {
		std::printf("  ok:   %s\n", what);
	}
}

TickEvent trade(int64_t ts_ms, uint64_t id, double px, double qty,
                double bid, double ask, int kind = 6)
{
	TickEvent e{};
	e.product_id = 1;
	e.update_kind = kind;
	e.has_trade = true;
	e.last_px = px;
	e.last_qty = qty;
	e.last_trade_ts_ms = static_cast<uint64_t>(ts_ms);
	e.last_trade_id = id;
	e.bid_px = bid;
	e.ask_px = ask;
	e.has_book = (bid > 0.0 || ask > 0.0);
	e.aggressor_is_buy = -1;   // matches the live feed: maker flag unpopulated
	e.recv_ts_ns = static_cast<uint64_t>(ts_ms) * 1'000'000ULL;
	return e;
}

std::vector<KlineBar> drain(KlineBuilder& b)
{
	std::vector<KlineBar> out;
	KlineBar k{};
	while (b.pop(&k)) out.push_back(k);
	return out;
}

constexpr int64_t P = 900'000;          // 15m in ms
constexpr int64_t T0 = 1'787'000'400'000; // an exact 15m boundary

void selftest()
{
	std::printf("[1] first bucket is discarded as partial\n");
	{
		KlineBuilder b(900, 1000);
		b.onTick(trade(T0 + 100, 1, 100.0, 1.0, 99.0, 101.0));
		b.onTick(trade(T0 + P + 10, 2, 200.0, 2.0, 199.0, 201.0));
		auto bars = drain(b);
		check(bars.empty(), "no bar emitted for the bucket we joined mid-way");
		check(b.partialBucketsDropped() == 1, "partial counted, not silently dropped");
	}

	std::printf("[2] a full bucket produces Binance's nine columns\n");
	{
		KlineBuilder b(900, 1000);
		b.onTick(trade(T0 + 1, 1, 10.0, 1.0, 9.0, 11.0));       // partial, discarded
		b.onTick(trade(T0 + P + 1, 2, 100.0, 1.0, 99.0, 101.0)); // opens bucket 1
		b.onTick(trade(T0 + P + 2, 3, 105.0, 2.0, 99.0, 101.0)); // >= ask -> buy
		b.onTick(trade(T0 + P + 3, 4,  98.0, 3.0, 99.0, 101.0)); // <= bid -> sell
		b.onTick(trade(T0 + 2 * P + 1, 5, 50.0, 1.0, 49.0, 51.0)); // closes bucket 1
		auto bars = drain(b);
		check(bars.size() == 1, "exactly one bar closed");
		if (bars.size() == 1) {
			const KlineBar& k = bars[0];
			check(k.bucket_open_ms == T0 + P, "bucket open stamped at the grid point");
			check(k.open == 100.0, "open = first trade of the bucket");
			check(k.high == 105.0, "high");
			check(k.low == 98.0, "low");
			check(k.close == 98.0, "close = last trade of the bucket");
			check(std::fabs(k.volume - 6.0) < 1e-12, "volume = sum(qty)");
			check(std::fabs(k.quote_volume - (100.0 + 210.0 + 294.0)) < 1e-9,
			      "quote_volume = sum(px*qty)");
			check(k.number_of_trades == 3, "number_of_trades");
			// quote rule: 100 is inside (99,101) -> unclassified; 105 >= ask -> buy;
			// 98 <= bid -> sell.
			check(std::fabs(k.taker_buy_base_volume - 2.0) < 1e-12,
			      "taker_buy_base = only the ask-crossing trade");
			check(k.n_trades_unclassified == 1,
			      "inside-spread trade counted unclassified, not guessed");
			check(k.aggressor_source == KlineBar::AggressorSource::QUOTE_RULE,
			      "provenance reported as approximate, not exact");
		}
	}

	std::printf("[3] empty buckets are emitted flat, one per bucket\n");
	{
		KlineBuilder b(900, 1000);
		b.onTick(trade(T0 + 1, 1, 10.0, 1.0, 9.0, 11.0));         // partial
		b.onTick(trade(T0 + P + 1, 2, 100.0, 1.0, 99.0, 101.0));  // bucket 1
		b.onTick(trade(T0 + 4 * P + 1, 3, 120.0, 1.0, 119.0, 121.0)); // skips 2 and 3
		auto bars = drain(b);
		check(bars.size() == 3, "closed bucket 1 plus two flat gap bars");
		if (bars.size() == 3) {
			check(bars[1].volume == 0.0 && bars[1].number_of_trades == 0, "gap bar is empty");
			check(bars[1].open == 100.0 && bars[1].close == 100.0,
			      "gap bar carries the previous close on all four prices");
			check(bars[1].bucket_open_ms == T0 + 2 * P
			      && bars[2].bucket_open_ms == T0 + 3 * P, "one bar per skipped bucket");
		}
	}

	std::printf("[4] aggTrade is counted but never applied (no double count)\n");
	{
		KlineBuilder b(900, 1000);
		b.onTick(trade(T0 + 1, 1, 10.0, 1.0, 9.0, 11.0));
		b.onTick(trade(T0 + P + 1, 2, 100.0, 1.0, 99.0, 101.0));
		b.onTick(trade(T0 + P + 2, 0, 100.0, 5.0, 99.0, 101.0, 7)); // aggTrade
		b.onTick(trade(T0 + 2 * P + 1, 3, 50.0, 1.0, 49.0, 51.0));
		auto bars = drain(b);
		check(bars.size() == 1 && std::fabs(bars[0].volume - 1.0) < 1e-12,
		      "aggTrade volume excluded");
		check(bars.size() == 1 && bars[0].number_of_trades == 1, "aggTrade not counted as a trade");
		check(b.aggTradeUpdates() == 1, "but it IS visible in the diagnostics");
	}

	std::printf("[5] duplicate and late trades are dropped and counted\n");
	{
		KlineBuilder b(900, 1000);
		b.onTick(trade(T0 + 1, 1, 10.0, 1.0, 9.0, 11.0));
		b.onTick(trade(T0 + P + 1, 5, 100.0, 1.0, 99.0, 101.0));
		b.onTick(trade(T0 + P + 2, 5, 100.0, 1.0, 99.0, 101.0));  // same id
		b.onTick(trade(T0 + P + 3, 4, 100.0, 1.0, 99.0, 101.0));  // older id
		b.onTick(trade(T0 + 2 * P + 1, 6, 50.0, 1.0, 49.0, 51.0));
		auto bars = drain(b);
		check(b.duplicatesDropped() == 2, "both replays dropped");
		check(bars.size() == 1 && bars[0].number_of_trades == 1, "bar counts the trade once");
	}

	std::printf("[6] a trade exactly on the boundary opens the NEW bucket\n");
	{
		KlineBuilder b(900, 1000);
		b.onTick(trade(T0 + 1, 1, 10.0, 1.0, 9.0, 11.0));
		b.onTick(trade(T0 + P, 2, 100.0, 1.0, 99.0, 101.0));   // exactly at open
		b.onTick(trade(T0 + 2 * P, 3, 50.0, 1.0, 49.0, 51.0)); // exactly at next open
		auto bars = drain(b);
		check(bars.size() == 1 && bars[0].bucket_open_ms == T0 + P,
		      "boundary trade belongs to the bucket it opens");
		check(bars.size() == 1 && bars[0].close == 100.0,
		      "the next boundary trade is NOT included in the closing bar");
	}

	std::printf("[7] backfill validation refuses gaps and off-grid bars\n");
	{
		KlineBuilder b(900, 1000);
		KlineBar good[3]{};
		for (int i = 0; i < 3; ++i) { good[i].bucket_open_ms = T0 + i * P; good[i].close = 10.0 + i; }
		check(b.ingestBackfill(good, 3), "contiguous on-grid run accepted");

		KlineBuilder b2(900, 1000);
		KlineBar gap[2]{};
		gap[0].bucket_open_ms = T0;
		gap[1].bucket_open_ms = T0 + 2 * P;   // hole
		check(!b2.ingestBackfill(gap, 2), "gap refused");

		KlineBuilder b3(900, 1000);
		KlineBar off[1]{};
		off[0].bucket_open_ms = T0 + 7;       // off grid
		check(!b3.ingestBackfill(off, 1), "off-grid bucket refused");
	}

	std::printf("[8] the backfill->live seam leaves no silent hole\n");
	{
		// The regression this suite missed: backfill ends at B, the process
		// attaches mid-way through B+1 (discarded as partial), so the first
		// built bar is B+2 and bucket B+1 was simply absent from the series
		// with nothing reporting it.
		KlineBuilder b(900, 1000);
		KlineBar bf[4]{};
		for (int i = 0; i < 4; ++i) { bf[i].bucket_open_ms = T0 + i * P; bf[i].close = 100.0 + i; }
		check(b.ingestBackfill(bf, 4), "backfill of 4 contiguous bars accepted");
		check(b.contiguousBars() == 4, "contiguous run is 4, and it is RETAINED");

		b.onTick(trade(T0 + 4 * P + 420000, 1, 200.0, 1.0, 199.0, 201.0)); // mid-bucket
		b.onTick(trade(T0 + 5 * P + 1000, 2, 210.0, 1.0, 209.0, 211.0));   // closes B+1 (partial)
		b.onTick(trade(T0 + 6 * P + 1000, 3, 220.0, 1.0, 219.0, 221.0));   // closes B+2
		auto bars = drain(b);
		check(bars.size() == 1 && bars[0].bucket_open_ms == T0 + 5 * P,
		      "first built bar is B+2, one bucket past the backfill");
		check(b.seamGaps() == 1, "the hole is DETECTED, not silent");
		check(b.lastGapFromMs() == T0 + 4 * P && b.lastGapToMs() == T0 + 4 * P,
		      "the missing bucket range is reported exactly");
		check(b.contiguousBars() == 1,
		      "warmth falls back to the post-gap run — 5 bars with a hole are not 5 bars");

		// And the hole can be closed without fabricating the partial bucket.
		KlineBar fill[1]{};
		fill[0].bucket_open_ms = T0 + 4 * P;
		fill[0].close = 205.0;
		check(b.ingestBackfill(fill, 1), "prepending the missing bucket is accepted");
		// WAS 2, and 2 WAS THE BUG (finding B1): the gap had CLEARED the four
		// backfill bars, so closing the hole could only ever recover the fill
		// plus the one live bar, and every start fell back to a live warmup.
		// The four are now quarantined and spliced back, so the run is 6.
		check(b.contiguousBars() == 6,
		      "history is contiguous again across the seam AND the backfill survived");
	}

	std::printf("[9] backfill that overlaps or leaves a hole is refused\n");
	{
		KlineBuilder b(900, 1000);
		KlineBar a[2]{};
		a[0].bucket_open_ms = T0; a[1].bucket_open_ms = T0 + P;
		check(b.ingestBackfill(a, 2), "first run accepted");
		check(b.ingestBackfill(a, 2) == false,
		      "re-ingesting the SAME run is refused (previously double-counted)");
		KlineBar hole[1]{};
		hole[0].bucket_open_ms = T0 + 3 * P;   // skips T0+2P
		check(b.ingestBackfill(hole, 1) == false, "run that would leave a hole is refused");
		KlineBar next[1]{};
		next[0].bucket_open_ms = T0 + 2 * P;
		check(b.ingestBackfill(next, 1), "the genuinely adjacent run appends");
		check(b.contiguousBars() == 3, "and the run is 3");
	}

	std::printf("[10] history is capped at max_history\n");
	{
		KlineBuilder b(900, 3);
		KlineBar bf[6]{};
		for (int i = 0; i < 6; ++i) { bf[i].bucket_open_ms = T0 + i * P; bf[i].close = 1.0 + i; }
		check(b.ingestBackfill(bf, 6), "6 bars ingested into a 3-bar ring");
		check(b.contiguousBars() == 3, "ring holds the newest 3");
		check(b.newestBar() != nullptr && b.newestBar()->bucket_open_ms == T0 + 5 * P,
		      "and the newest is the newest");
	}

	// ---- finding B1: the boot seam, end to end --------------------------
	//
	// The production failure this closes:
	//     [AGDIAG] bars_seen=713 contiguous=14/700 backfilled=699 seam_gaps=1
	// 699 backfill bars were ingested and then thrown away by the very first
	// built bar, every start, because the one bucket between them is missing
	// BY CONSTRUCTION: fetch_binance_klines.py can only write CLOSED buckets,
	// and the bucket the process attaches to is discarded as a partial. The
	// bar layer cannot invent that bucket — it saw only part of it — so the
	// fix is to keep the 699 alive until the bucket is fetched and spliced in.

	std::printf("[11] boot seam: the 699-bar backfill SURVIVES and repairs to warm\n");
	{
		constexpr int kWarm = 700;
		constexpr int kBackfill = 699;
		const int64_t B = T0 + kBackfill * P;   // the bucket we attach mid-way through

		KlineBuilder b(900, kWarm);
		std::vector<KlineBar> bf(kBackfill);
		for (int i = 0; i < kBackfill; ++i) {
			bf[i].bucket_open_ms = T0 + static_cast<int64_t>(i) * P;   // ends at B - 1
			bf[i].close = 100.0 + i;
			bf[i].from_backfill = true;
		}
		check(b.ingestBackfill(bf.data(), kBackfill), "699 closed bars accepted at boot");
		check(b.contiguousBars() == kBackfill, "699 contiguous, one short of warm");

		// Attach mid-bucket B, then let B+1 and B+2 close normally.
		b.onTick(trade(B + 420000, 1, 200.0, 1.0, 199.0, 201.0));
		b.onTick(trade(B + P + 1000, 2, 210.0, 1.0, 209.0, 211.0));
		b.onTick(trade(B + 2 * P + 1000, 3, 220.0, 1.0, 219.0, 221.0));
		auto bars = drain(b);
		check(bars.size() == 1 && bars[0].bucket_open_ms == B + P,
		      "first EMITTED bar is B+1; bucket B is never emitted");
		check(b.partialBucketsDropped() == 1, "bucket B discarded as partial, not fabricated");

		// The regression: this used to be 1 (everything before the seam gone).
		check(b.pendingBars() == kBackfill,
		      "all 699 backfill bars are QUARANTINED, not discarded");
		check(b.contiguousBars() == 1,
		      "and warmth stays honest at 1 while the hole is open");
		check(b.missingFromMs() == B && b.missingToMs() == B,
		      "the outstanding hole is reported as exactly bucket B");

		// A CSV refreshed after startup contains B, because B has closed.
		KlineBar fill[1]{};
		fill[0].bucket_open_ms = B;
		fill[0].close = 12345.0;          // distinctive: NOT the partial's 200/210
		fill[0].volume = 77.0;
		fill[0].from_backfill = true;
		check(b.ingestBackfill(fill, 1), "the refreshed CSV's bucket B is accepted");
		check(b.seamRepairs() == 1, "the quarantine was spliced back, and says so");
		check(b.pendingBars() == 0, "nothing left quarantined");
		check(b.contiguousBars() == kWarm,
		      "700 CONTIGUOUS bars — warm on the first bar after the fill, not in 7.3 days");
		check(b.missingFromMs() == 0 && b.missingToMs() == 0, "no hole outstanding");

		// And bucket B holds the REST kline, never a synthesised bar.
		const KlineBar* fixed = b.barAt(b.contiguousBars() - 2);
		check(fixed != nullptr && fixed->bucket_open_ms == B,
		      "bucket B sits in the window in its right place");
		check(fixed != nullptr && fixed->from_backfill && fixed->close == 12345.0
		      && fixed->volume == 77.0,
		      "bucket B is the INGESTED kline, not the partial we half-observed");
		const KlineBar* newest = b.newestBar();
		check(newest != nullptr && newest->bucket_open_ms == B + P && !newest->from_backfill,
		      "and the bar after it is the live-built one");
	}

	std::printf("[12] a STALE csv leaves a multi-bucket hole; a partial fill is refused\n");
	{
		// Restart with a CSV written five buckets ago (fetched, then the launch
		// took a while). The hole is [B-4 .. B], not just B.
		KlineBuilder b(900, 1000);
		KlineBar bf[6]{};
		for (int i = 0; i < 6; ++i) { bf[i].bucket_open_ms = T0 + i * P; bf[i].close = 10.0 + i; }
		check(b.ingestBackfill(bf, 6), "stale backfill ending 5 buckets back accepted");

		const int64_t B = T0 + 10 * P;
		b.onTick(trade(B + 60000, 1, 50.0, 1.0, 49.0, 51.0));      // attach mid-B
		b.onTick(trade(B + P + 1000, 2, 51.0, 1.0, 50.0, 52.0));
		b.onTick(trade(B + 2 * P + 1000, 3, 52.0, 1.0, 51.0, 53.0));
		(void)drain(b);
		check(b.missingFromMs() == T0 + 6 * P && b.missingToMs() == B,
		      "the whole stale stretch is reported, not just the attach bucket");
		check(b.pendingBars() == 6, "and the stale backfill is still held");

		// Four of the five missing buckets: still a hole, so refused outright.
		KlineBar shortfill[4]{};
		for (int i = 0; i < 4; ++i) shortfill[i].bucket_open_ms = T0 + (6 + i) * P;
		check(!b.ingestBackfill(shortfill, 4), "a fill that does not close the hole is REFUSED");
		check(b.pendingBars() == 6 && b.contiguousBars() == 1,
		      "and it changed nothing — a refused ingest ingests nothing");

		KlineBar full[5]{};
		for (int i = 0; i < 5; ++i) full[i].bucket_open_ms = T0 + (6 + i) * P;
		check(b.ingestBackfill(full, 5), "the complete 5-bucket fill is accepted");
		check(b.contiguousBars() == 12, "6 backfill + 5 fill + 1 live = 12 contiguous");
		check(b.seamGaps() == 1 && b.seamRepairs() == 1, "one hole, one repair");
	}

	std::printf("[13] an illiquid symbol: B+1 empty too, and no SECOND hole opens\n");
	{
		// Nothing trades for two whole buckets after the one we attached on.
		// Those buckets are genuinely observed as empty (we were connected), so
		// they are flat bars — that is rule 4, not fabrication. Only bucket B,
		// which we saw part of, is missing.
		KlineBuilder b(900, 1000);
		KlineBar bf[4]{};
		for (int i = 0; i < 4; ++i) { bf[i].bucket_open_ms = T0 + i * P; bf[i].close = 100.0 + i; }
		check(b.ingestBackfill(bf, 4), "backfill accepted");

		const int64_t B = T0 + 4 * P;
		b.onTick(trade(B + 300000, 1, 200.0, 1.0, 199.0, 201.0));   // attach mid-B
		b.onTick(trade(B + 3 * P + 1000, 2, 205.0, 1.0, 204.0, 206.0)); // B+1,B+2 silent
		b.onTick(trade(B + 4 * P + 1000, 3, 206.0, 1.0, 205.0, 207.0));
		auto bars = drain(b);
		check(bars.size() == 3 && bars[0].bucket_open_ms == B + P
		      && bars[0].number_of_trades == 0,
		      "B+1 is an EMPTY bar, emitted flat, not skipped");
		check(b.seamGaps() == 1 && b.missingFromMs() == B && b.missingToMs() == B,
		      "silence does not widen the hole: only bucket B is missing");
		check(b.quarantinesDiscarded() == 0,
		      "and no second discontinuity opened — empty buckets are filled, not gapped");
		check(b.pendingBars() == 4, "the backfill is intact through the quiet stretch");

		KlineBar fill[1]{};
		fill[0].bucket_open_ms = B;
		fill[0].close = 199.5;
		check(b.ingestBackfill(fill, 1), "bucket B fill accepted");
		check(b.contiguousBars() == 8, "4 backfill + B + B+1..B+3 = 8 contiguous");
	}

	std::printf("[14] a bar NOT newer than history is a rewind: dropped, counted, no hole\n");
	{
		// A feed lagging the CSV (or a wrong clock on either side). The old
		// code called this a discontinuity and reported a BACKWARDS missing
		// range, which no operator could act on.
		KlineBuilder b(900, 1000);
		KlineBar bf[6]{};
		for (int i = 0; i < 6; ++i) { bf[i].bucket_open_ms = T0 + i * P; bf[i].close = 10.0 + i; }
		check(b.ingestBackfill(bf, 6), "backfill through T0+5P accepted");

		b.onTick(trade(T0 + 2 * P + 100, 1, 50.0, 1.0, 49.0, 51.0));   // attach WAY behind
		b.onTick(trade(T0 + 3 * P + 100, 2, 51.0, 1.0, 50.0, 52.0));   // drops the partial
		b.onTick(trade(T0 + 4 * P + 100, 3, 52.0, 1.0, 51.0, 53.0));   // emits T0+3P
		auto bars = drain(b);
		check(bars.size() == 1 && bars[0].bucket_open_ms == T0 + 3 * P,
		      "the stale bar is still EMITTED, so the log shows it");
		check(b.rewoundBarsDropped() == 1, "but it is refused entry to the window, and counted");
		check(b.seamGaps() == 0 && b.missingFromMs() == 0,
		      "and it is NOT reported as a hole with a backwards range");
		check(b.contiguousBars() == 6 && b.newestBar()->bucket_open_ms == T0 + 5 * P,
		      "the retained window is untouched — the past is never rewritten");
	}

	std::printf("[15] repair works when the ring is already FULL (trim must not eat the fill)\n");
	{
		// max_history == the backfill length: prepending the fill overflows the
		// ring, and trimming before splicing would pop exactly the bar just
		// added, leaving the hole open with ingestBackfill() reporting success.
		constexpr int kRing = 8;
		KlineBuilder b(900, kRing);
		KlineBar bf[kRing]{};
		for (int i = 0; i < kRing; ++i) { bf[i].bucket_open_ms = T0 + i * P; bf[i].close = 5.0 + i; }
		check(b.ingestBackfill(bf, kRing), "ring filled exactly by the backfill");

		const int64_t B = T0 + kRing * P;
		b.onTick(trade(B + 1000, 1, 90.0, 1.0, 89.0, 91.0));
		b.onTick(trade(B + P + 1000, 2, 91.0, 1.0, 90.0, 92.0));
		b.onTick(trade(B + 2 * P + 1000, 3, 92.0, 1.0, 91.0, 93.0));
		(void)drain(b);
		check(b.pendingBars() == kRing && b.contiguousBars() == 1, "full ring quarantined");

		KlineBar fill[1]{};
		fill[0].bucket_open_ms = B;
		fill[0].close = 90.5;
		check(b.ingestBackfill(fill, 1), "fill accepted into a full ring");
		check(b.contiguousBars() == kRing,
		      "the ring is contiguous and full — the trim did not eat the fill");
		check(b.newestBar()->bucket_open_ms == B + P && b.missingFromMs() == 0,
		      "and it holds the NEWEST kRing buckets, hole closed");
	}
}

// ---- replay mode ---------------------------------------------------------

std::vector<std::string> split(const std::string& s, char d)
{
	std::vector<std::string> out;
	std::stringstream ss(s);
	std::string item;
	while (std::getline(ss, item, d)) out.push_back(item);
	return out;
}

int replay(const char* tick_csv, const char* kline_csv)
{
	std::ifstream tf(tick_csv);
	if (!tf) { std::printf("cannot open %s\n", tick_csv); return 2; }

	KlineBuilder b(900, 1000);
	std::vector<KlineBar> built;
	std::string line;
	std::getline(tf, line);   // header
	while (std::getline(tf, line)) {
		if (line.empty()) continue;
		auto f = split(line, ',');
		if (f.size() < 7) continue;
		TickEvent e = trade(std::stoll(f[0]), std::stoull(f[1]), std::stod(f[2]),
		                    std::stod(f[3]), std::stod(f[4]), std::stod(f[5]),
		                    std::stoi(f[6]));
		b.onTick(e);
		KlineBar k{};
		while (b.pop(&k)) built.push_back(k);
	}

	std::map<int64_t, std::vector<double>> ref;
	std::ifstream kf(kline_csv);
	if (!kf) { std::printf("cannot open %s\n", kline_csv); return 2; }
	std::getline(kf, line);
	while (std::getline(kf, line)) {
		if (line.empty()) continue;
		auto f = split(line, ',');
		if (f.size() < 10) continue;
		ref[std::stoll(f[0])] = {std::stod(f[1]), std::stod(f[2]), std::stod(f[3]),
		                         std::stod(f[4]), std::stod(f[5]), std::stod(f[6]),
		                         std::stod(f[7]), std::stod(f[8]), std::stod(f[9])};
	}

	const char* names[9] = {"open", "high", "low", "close", "volume",
	                        "quote_volume", "n_trades", "tb_base", "tb_quote"};
	int match[9] = {0};
	int compared = 0;
	double worst[9] = {0};

	for (const auto& k : built) {
		auto it = ref.find(k.bucket_open_ms);
		if (it == ref.end()) continue;
		++compared;
		const double got[9] = {k.open, k.high, k.low, k.close, k.volume, k.quote_volume,
		                       static_cast<double>(k.number_of_trades),
		                       k.taker_buy_base_volume, k.taker_buy_quote_volume};
		for (int i = 0; i < 9; ++i) {
			const double a = got[i], e = it->second[i];
			// Relative tolerance: both sides accumulate the same trades into a
			// double in a different order, so identical values still differ in
			// the last ULP. Exact equality is the wrong test for summed columns.
			const double denom = std::max(1.0, std::fabs(e));
			const double rel = std::fabs(a - e) / denom;
			if (rel <= 1e-9) ++match[i];
			if (rel > worst[i]) worst[i] = rel;
		}
	}

	std::printf("\ncompared %d bars built from ticks against Binance's own klines\n", compared);
	std::printf("%-14s %10s %14s\n", "column", "match", "worst rel err");
	for (int i = 0; i < 9; ++i) {
		std::printf("%-14s %6d/%-4d %14.3e\n", names[i], match[i], compared, worst[i]);
	}
	std::printf("\ndiagnostics: trade_upd=%lld aggtrade_upd=%lld dup_dropped=%lld"
	            " late_dropped=%lld partial_dropped=%lld\n",
	            (long long)b.tradeUpdates(), (long long)b.aggTradeUpdates(),
	            (long long)b.duplicatesDropped(), (long long)b.lateTradesDropped(),
	            (long long)b.partialBucketsDropped());
	return 0;
}

} // namespace

int main(int argc, char** argv)
{
	const char* ticks = nullptr;
	const char* klines = nullptr;
	bool self = false;
	for (int i = 1; i < argc; ++i) {
		if (!std::strcmp(argv[i], "--selftest")) self = true;
		else if (!std::strcmp(argv[i], "--ticks") && i + 1 < argc) ticks = argv[++i];
		else if (!std::strcmp(argv[i], "--klines") && i + 1 < argc) klines = argv[++i];
	}
	if (self) {
		selftest();
		std::printf("\n%s (%d failure%s)\n", g_failures ? "FAILED" : "PASSED",
		            g_failures, g_failures == 1 ? "" : "s");
		return g_failures ? 1 : 0;
	}
	if (ticks && klines) return replay(ticks, klines);
	std::printf("usage: %s --selftest | --ticks T.csv --klines K.csv\n", argv[0]);
	return 2;
}
