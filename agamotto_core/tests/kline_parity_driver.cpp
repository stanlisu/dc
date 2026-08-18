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
		KlineBuilder b(900);
		b.onTick(trade(T0 + 100, 1, 100.0, 1.0, 99.0, 101.0));
		b.onTick(trade(T0 + P + 10, 2, 200.0, 2.0, 199.0, 201.0));
		auto bars = drain(b);
		check(bars.empty(), "no bar emitted for the bucket we joined mid-way");
		check(b.partialBucketsDropped() == 1, "partial counted, not silently dropped");
	}

	std::printf("[2] a full bucket produces Binance's nine columns\n");
	{
		KlineBuilder b(900);
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
		KlineBuilder b(900);
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
		KlineBuilder b(900);
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
		KlineBuilder b(900);
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
		KlineBuilder b(900);
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
		KlineBuilder b(900);
		KlineBar good[3]{};
		for (int i = 0; i < 3; ++i) { good[i].bucket_open_ms = T0 + i * P; good[i].close = 10.0 + i; }
		check(b.ingestBackfill(good, 3), "contiguous on-grid run accepted");

		KlineBuilder b2(900);
		KlineBar gap[2]{};
		gap[0].bucket_open_ms = T0;
		gap[1].bucket_open_ms = T0 + 2 * P;   // hole
		check(!b2.ingestBackfill(gap, 2), "gap refused");

		KlineBuilder b3(900);
		KlineBar off[1]{};
		off[0].bucket_open_ms = T0 + 7;       // off grid
		check(!b3.ingestBackfill(off, 1), "off-grid bucket refused");
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

	KlineBuilder b(900);
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
