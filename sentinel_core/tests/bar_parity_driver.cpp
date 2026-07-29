// Parity driver: replay an event stream through the C++ BarBuilder and print
// every emitted bar as CSV. The Python reference replays the SAME file and
// prints the same CSV; any field-level difference is a parity break.
//
// Event CSV on stdin (one per line):
//   T,ts_ms,price,qty,is_buyer_maker,n_trades
//   B,ts_ms,bid_p,bid_q,ask_p,ask_q
//   D,ts_ms,bp0..bp4,bq0..bq4,ap0..ap4,aq0..aq4
//   M,ts_ms,mark,index,funding,predicted_funding
//   L,ts_ms,is_buy,notional
//   O,ts_ms,oi
#include "../src/bar_builder.hpp"

#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace mjolnir;

namespace {

std::vector<std::string> split(const std::string& s, char d)
{
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, d)) out.push_back(item);
    return out;
}

void printHeader()
{
    std::printf("bucket_ms,open,high,low,close,volume,buy_vol,sell_vol,n_trades,vwap,"
                "trade_imbalance,bid_price,bid_amount,ask_price,ask_amount,");
    for (int i = 0; i < BOOK_LEVELS; ++i)
        std::printf("bids_%d_price,bids_%d_qty,asks_%d_price,asks_%d_qty,", i, i, i, i);
    std::printf("depth_bid_L1,depth_bid_L3,depth_bid_L5,depth_ask_L1,depth_ask_L3,depth_ask_L5,"
                "mark_price,index_price,funding_rate,predicted_funding_rate,open_interest,"
                "liq_long_notional,liq_short_notional,liq_long_count,liq_short_count,"
                "liq_total_count,cycle_progress,secs_to_boundary\n");
}

void printBar(const Bar& b)
{
    // %.17g round-trips a double exactly, so the comparison tests the MATH and
    // not the formatting.
    std::printf("%lld,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%lld,%.17g,%.17g,"
                "%.17g,%.17g,%.17g,%.17g,",
                (long long)b.bucket_ms, b.open, b.high, b.low, b.close, b.volume,
                b.buy_vol, b.sell_vol, (long long)b.n_trades, b.vwap, b.trade_imbalance,
                b.bid_price, b.bid_amount, b.ask_price, b.ask_amount);
    for (int i = 0; i < BOOK_LEVELS; ++i)
        std::printf("%.17g,%.17g,%.17g,%.17g,", b.bids_price[i], b.bids_qty[i],
                    b.asks_price[i], b.asks_qty[i]);
    std::printf("%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
                "%.17g,%.17g,%lld,%lld,%lld,%.17g,%lld\n",
                b.depth_bid_L1, b.depth_bid_L3, b.depth_bid_L5,
                b.depth_ask_L1, b.depth_ask_L3, b.depth_ask_L5,
                b.mark_price, b.index_price, b.funding_rate, b.predicted_funding_rate,
                b.open_interest, b.liq_long_notional, b.liq_short_notional,
                (long long)b.liq_long_count, (long long)b.liq_short_count,
                (long long)b.liq_total_count, b.cycle_progress,
                (long long)b.secs_to_boundary);
}

} // namespace

int main(int argc, char** argv)
{
    int bar_sec = 5, target_sec = 5;
    if (argc >= 2) bar_sec = std::stoi(argv[1]);
    if (argc >= 3) target_sec = std::stoi(argv[2]);

    BarBuilder bb(bar_sec, target_sec);
    printHeader();

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        auto f = split(line, ',');
        const std::string& t = f[0];
        const int64_t ts = std::stoll(f[1]);

        if (t == "T") {
            Bar out;
            if (bb.onTrade(std::stod(f[2]), std::stod(f[3]), std::stoi(f[4]) != 0, ts,
                           std::stoll(f[5]), &out)) {
                printBar(out);
            }
        } else if (t == "B") {
            bb.onBookTicker(std::stod(f[2]), std::stod(f[3]), std::stod(f[4]), std::stod(f[5]), ts);
        } else if (t == "D") {
            double bp[5], bq[5], ap[5], aq[5];
            for (int i = 0; i < 5; ++i) {
                bp[i] = std::stod(f[2 + i]);
                bq[i] = std::stod(f[7 + i]);
                ap[i] = std::stod(f[12 + i]);
                aq[i] = std::stod(f[17 + i]);
            }
            bb.onDepth(bp, bq, ap, aq, 5, ts);
        } else if (t == "M") {
            bb.onMarkPrice(std::stod(f[2]), std::stod(f[3]), std::stod(f[4]), std::stod(f[5]), ts);
        } else if (t == "L") {
            bb.onLiquidation(std::stoi(f[2]) != 0, std::stod(f[3]), ts);
        } else if (t == "O") {
            bb.setOpenInterest(std::stod(f[2]), ts);
        } else {
            std::fprintf(stderr, "unknown event type: %s\n", t.c_str());
            return 2;
        }
    }
    return 0;
}
