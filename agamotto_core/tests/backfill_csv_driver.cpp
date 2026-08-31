// Backfill-CSV parser — the gate for sentinel's venue-kline read path.
//
// WHY A PURE PARSER, TESTED HERE. AgamottoStrategy cannot be linked in a test
// binary at all: AlgoBase has an out-of-line ctor and send_order, and
// ltp_strat_sdk/lib/ is gitignored. The parser used to be a member function of
// the strategy, which meant its only test was a live venue -- and it feeds three
// paths that all fail closed: loadBackfill() HALTS the run on a bad parse,
// repairSeam() cannot close a boot-seam hole without it, and
// reconcileFromBackfill() is the only thing that repairs bars the SHM feed
// under-delivered.
//
// So backfill_csv.hpp is <cerrno>/<cstdlib>/<vector> + agamotto_core.hpp only,
// and every rule below has a mutant in run_backfill_csv_mutants.sh.
//
// THE CHECK THAT ACTUALLY MATTERS is the last one: a DIFFERENTIAL against the
// old std::istringstream body, reproduced verbatim in refParse() below, over
// whatever real fleet CSVs are handed to it. The rewrite's whole claim is "same
// bars, less stall on the ring-draining thread", and unit checks on synthetic
// rows cannot prove the first half of that. Point it at hydra's config/ dir:
//
//     ./drv /path/to/agamotto_test/config
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "backfill_csv.hpp"

using agamotto::KlineBar;
using namespace agamotto::backfill;

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

constexpr int kBarSec = 900;   // 15m, the deployed fleet's period

bool parse(const std::string& csv, std::vector<KlineBar>& out, ParseError& err)
{
    return parseRows(csv.data(), csv.size(), kBarSec, out, err);
}

// ---------------------------------------------------------------------------
// THE OLD BODY, VERBATIM. Reproduced from AgamottoStrategy::parseBackfillCsv as
// it stood at sentinel 0c7b919, with only the ifstream swapped for an istream
// over the same bytes and the LOG_ERROR calls dropped. Do not "clean this up":
// its value is being an independent second implementation, and every edit that
// makes it prettier makes it less independent.
bool refParse(const std::string& csv, int barSec, std::vector<KlineBar>& out)
{
    out.clear();
    std::istringstream f_(csv);

    std::string line_;
    std::getline(f_, line_);   // header
    int lineNo_ = 1;
    (void)lineNo_;   // the old body used it only in the LOG_ERROR text
    while (std::getline(f_, line_)) {
        ++lineNo_;
        if (line_.empty()) continue;
        std::vector<std::string> fld_;
        std::string cell_;
        std::istringstream ls_(line_);
        while (std::getline(ls_, cell_, ',')) fld_.push_back(cell_);
        if (fld_.size() < 10) return false;
        KlineBar b{};
        try {
            b.bucket_open_ms = std::stoll(fld_[0]);
            b.open = std::stod(fld_[1]);
            b.high = std::stod(fld_[2]);
            b.low = std::stod(fld_[3]);
            b.close = std::stod(fld_[4]);
            b.volume = std::stod(fld_[5]);
            b.quote_volume = std::stod(fld_[6]);
            b.number_of_trades = std::stoll(fld_[7]);
            b.taker_buy_base_volume = std::stod(fld_[8]);
            b.taker_buy_quote_volume = std::stod(fld_[9]);
        } catch (const std::exception&) {
            return false;
        }
        b.bucket_close_ms = b.bucket_open_ms + static_cast<int64_t>(barSec) * 1000 - 1;
        b.aggressor_source = KlineBar::AggressorSource::EXACT_MAKER_FLAG;
        b.from_backfill = true;
        out.push_back(b);
    }
    if (out.empty()) return false;
    return true;
}

// BIT-identical, not close-enough. Both sides route the same decimal text
// through strtod (std::stod calls it), so any difference is a parsing defect,
// never a rounding one -- and a tolerance here would hide exactly the class of
// bug this comparison exists to find.
bool sameBar(const KlineBar& a, const KlineBar& b)
{
    return a.bucket_open_ms == b.bucket_open_ms
        && a.bucket_close_ms == b.bucket_close_ms
        && std::memcmp(&a.open, &b.open, sizeof(double)) == 0
        && std::memcmp(&a.high, &b.high, sizeof(double)) == 0
        && std::memcmp(&a.low, &b.low, sizeof(double)) == 0
        && std::memcmp(&a.close, &b.close, sizeof(double)) == 0
        && std::memcmp(&a.volume, &b.volume, sizeof(double)) == 0
        && std::memcmp(&a.quote_volume, &b.quote_volume, sizeof(double)) == 0
        && a.number_of_trades == b.number_of_trades
        && std::memcmp(&a.taker_buy_base_volume, &b.taker_buy_base_volume,
                       sizeof(double)) == 0
        && std::memcmp(&a.taker_buy_quote_volume, &b.taker_buy_quote_volume,
                       sizeof(double)) == 0
        && a.aggressor_source == b.aggressor_source
        && a.from_backfill == b.from_backfill;
}

const char* kHeader =
    "open_ms,open,high,low,close,volume,quote_volume,n_trades,"
    "taker_buy_base,taker_buy_quote\n";

// A real BTCUSDT 15m row, copied off hydra.
const char* kRow =
    "1788137100000,108451.10,108512.30,108402.00,108498.70,"
    "312.44500000,33887412.19500000,18432,161.20100000,17486221.03100000\n";

}  // namespace

int main(int argc, char** argv)
{
    // -- the golden row, field by field ------------------------------------
    {
        std::vector<KlineBar> bars;
        ParseError err{};
        check(parse(std::string(kHeader) + kRow, bars, err), "golden row parses");
        check(bars.size() == 1, "golden row yields exactly one bar");
        if (bars.size() == 1) {
            const KlineBar& b = bars[0];
            check(b.bucket_open_ms == 1788137100000LL, "open_ms");
            check(b.open == 108451.10, "open");
            check(b.high == 108512.30, "high");
            check(b.low == 108402.00, "low");
            check(b.close == 108498.70, "close");
            check(b.volume == 312.445, "volume");
            check(b.quote_volume == 33887412.195, "quote_volume");
            check(b.number_of_trades == 18432, "n_trades");
            check(b.taker_buy_base_volume == 161.201, "taker_buy_base");
            check(b.taker_buy_quote_volume == 17486221.031, "taker_buy_quote");
            // DERIVED, never read from the file. bar_sec comes from config; a
            // parser that inferred it from row spacing would silently agree
            // with a mislabelled file.
            check(b.bucket_close_ms == 1788137100000LL + 900 * 1000 - 1,
                  "bucket_close_ms = open + bar_sec*1000 - 1");
            // The taker columns came from Binance, so they are EXACT. A bar
            // that claimed QUOTE_RULE here would let a consumer read
            // venue-exact data as this build's own approximation.
            check(b.aggressor_source == KlineBar::AggressorSource::EXACT_MAKER_FLAG,
                  "aggressor_source is EXACT_MAKER_FLAG");
            // from_backfill is what keeps a spliced bar distinguishable from a
            // built one; the core's reconcile and seam logic both key on it.
            check(b.from_backfill, "from_backfill is set");
        }
    }

    // -- the header is discarded unread ------------------------------------
    {
        std::vector<KlineBar> bars;
        ParseError err{};
        // Line 1 is skipped WITHOUT being examined, exactly as the old
        // getline-and-discard did. A parser that tried to parse it would reject
        // every real file, whose first line is column names.
        check(parse(std::string("this is not remotely a csv row\n") + kRow, bars, err),
              "line 1 is skipped without being examined");
        check(bars.size() == 1, "and only the real row survives");

        std::vector<KlineBar> b2;
        ParseError e2{};
        check(!parse(kHeader, b2, e2), "a header-only file is rejected");
        check(e2.kind == ParseError::Kind::EMPTY, "  ... as EMPTY");
        check(!parse("", b2, e2), "an empty file is rejected");
        check(e2.kind == ParseError::Kind::EMPTY, "  ... as EMPTY");
    }

    // -- line endings and blank lines --------------------------------------
    {
        std::string crlf(kHeader);
        crlf = crlf.substr(0, crlf.size() - 1) + "\r\n";
        std::string row(kRow);
        row = row.substr(0, row.size() - 1) + "\r\n";
        std::vector<KlineBar> bars;
        ParseError err{};
        check(parse(crlf + row, bars, err), "CRLF file parses");
        // The '\r' rides on the LAST field. Left in place it would sit between
        // the number and the field end -- harmless for strtod, but it is the
        // kind of byte that becomes a defect the moment anything gets stricter.
        check(bars.size() == 1 && bars[0].taker_buy_quote_volume == 17486221.031,
              "CRLF does not corrupt the last field");

        // Same hazard as the empty last field, reached through CRLF instead:
        // unstripped, a '\r' IS the last field's entire content, and '\r' is
        // leading whitespace to strtod. Without the strip this parses the next
        // row's open_ms into taker_buy_quote_volume and reports success.
        std::vector<KlineBar> crlfEmpty;
        ParseError ce{};
        check(!parse(crlf + "1788137100000,1,2,3,4,5,6,7,8,\r\n" + row,
                     crlfEmpty, ce),
              "a CRLF row with an empty last field is rejected, not read across");

        std::vector<KlineBar> b2;
        ParseError e2{};
        check(parse(std::string(kHeader) + "\n" + kRow + "\n", b2, e2),
              "blank lines are skipped");
        check(b2.size() == 1, "  ... and yield no bars of their own");

        std::string noTrailingNl(std::string(kHeader) + kRow);
        noTrailingNl.pop_back();
        std::vector<KlineBar> b3;
        ParseError e3{};
        check(parse(noTrailingNl, b3, e3), "a file with no trailing newline parses");
        check(b3.size() == 1, "  ... and its last row is not dropped");
    }

    // -- malformed rows are REJECTED, never half-read ----------------------
    {
        std::vector<KlineBar> bars;
        ParseError err{};
        // Nine fields. The old body's silent alternative -- reading what is
        // there and leaving the rest zero -- would put a bar with volume 0 into
        // the rolling window, which is a correction that CAUSES the defect it
        // is meant to repair.
        check(!parse(std::string(kHeader) + "1,2,3,4,5,6,7,8,9\n", bars, err),
              "a nine-field row is rejected");
        check(err.kind == ParseError::Kind::FIELD_COUNT, "  ... as FIELD_COUNT");
        check(err.fields == 9, "  ... reporting 9 fields");
        check(err.line == 2, "  ... on line 2, counting the header as line 1");

        check(!parse(std::string(kHeader) +
                     "1788137100000,x,2,3,4,5,6,7,8,9\n", bars, err),
              "a non-numeric field is rejected");
        check(err.kind == ParseError::Kind::UNPARSEABLE, "  ... as UNPARSEABLE");

        check(!parse(std::string(kHeader) +
                     "1788137100000,1,2,3,4,5,6,,8,9\n", bars, err),
              "an EMPTY field is rejected");

        // A garbage INTEGER field, which is a different reader from the double
        // one above and needs its own case -- open_ms and n_trades are the only
        // two that go through strtoll.
        check(!parse(std::string(kHeader) +
                     "notanumber,1,2,3,4,5,6,7,8,9\n", bars, err),
              "a non-numeric open_ms is rejected");
        check(!parse(std::string(kHeader) +
                     "1788137100000,1,2,3,4,5,6,nope,8,9\n", bars, err),
              "a non-numeric n_trades is rejected");

        // std::stoll threw out_of_range; strtoll merely sets ERANGE and returns
        // LLONG_MAX, which would land in the panel as a plausible timestamp.
        check(!parse(std::string(kHeader) +
                     "99999999999999999999999999,1,2,3,4,5,6,7,8,9\n", bars, err),
              "an out-of-range open_ms is rejected");

        // THE HAZARD THE NO-COPY READER INTRODUCES, and the reason the
        // zero-length guard in toI64/toF64 is load-bearing rather than
        // defensive. Fields are views into one contiguous buffer, and strtod
        // skips LEADING WHITESPACE -- of which '\n' and '\r' are both. So an
        // empty LAST field points at the newline, and a reader that got as far
        // as calling strtod would skip it and parse the FIRST NUMBER OF THE NEXT
        // ROW: a bar silently built from two different buckets' bytes. Only an
        // empty field in the last column can do this; a mid-line empty field
        // points at ',', which stops strtod cold.
        check(!parse(std::string(kHeader) +
                     "1788137100000,1,2,3,4,5,6,7,8,\n" + kRow, bars, err),
              "an empty LAST field is rejected, not read across the line break");

        // Line numbering has to survive skipped lines, or an error names a row
        // the operator cannot find in the file.
        check(!parse(std::string(kHeader) + kRow + "\n" + "1,2,3\n", bars, err),
              "a short row after a blank line is rejected");
        check(err.line == 4, "  ... reported on line 4, not 3");
    }

    // -- loose numeric semantics are PRESERVED -----------------------------
    {
        // std::stod ignores trailing characters, and loadBackfill() turns a
        // parse failure into mHalted. Tightening this would be a fleet that
        // refuses to BOOT on a quirk nobody has seen; the tightening is a
        // separate decision from this rewrite.
        std::vector<KlineBar> bars;
        ParseError err{};
        check(parse(std::string(kHeader) +
                    "1788137100000,108451.10junk,2,3,4,5,6,7,8,9\n", bars, err),
              "trailing characters after a number are IGNORED, as std::stod does");
        check(bars.size() == 1 && bars[0].open == 108451.10,
              "  ... and the leading number is what is kept");

        std::vector<KlineBar> b2;
        ParseError e2{};
        check(!parse(std::string(kHeader) +
                     "1788137100000,1e999,2,3,4,5,6,7,8,9\n", b2, e2),
              "an out-of-range value is rejected, as std::stod's out_of_range is");
        check(e2.kind == ParseError::Kind::UNPARSEABLE, "  ... as UNPARSEABLE");
    }

    // -- extra columns are tolerated ---------------------------------------
    {
        std::vector<KlineBar> bars;
        ParseError err{};
        std::string wide(kRow);
        wide.pop_back();
        wide += ",99,100\n";
        check(parse(std::string(kHeader) + wide, bars, err),
              "a row with MORE than ten fields parses");
        check(bars.size() == 1 && bars[0].taker_buy_quote_volume == 17486221.031,
              "  ... reading the first ten, as the old body did");
    }

    // -- multi-row ordering ------------------------------------------------
    {
        std::string csv(kHeader);
        for (int i = 0; i < 5; ++i) {
            char row[256];
            std::snprintf(row, sizeof(row),
                          "%lld,1,2,3,4,5,6,7,8,9\n",
                          1788137100000LL + static_cast<long long>(i) * 900000LL);
            csv += row;
        }
        std::vector<KlineBar> bars;
        ParseError err{};
        check(parse(csv, bars, err), "a five-row file parses");
        check(bars.size() == 5, "  ... yielding five bars");
        bool ordered = bars.size() == 5;
        for (size_t i = 1; i < bars.size(); ++i) {
            if (bars[i].bucket_open_ms <= bars[i - 1].bucket_open_ms) ordered = false;
        }
        // File order is preserved as-is. The CORE validates grid alignment,
        // ordering and contiguity and ingests nothing on violation; the parser
        // must not quietly sort, or the core's refusal never fires.
        check(ordered, "  ... in file order, unsorted and unreordered");
    }

    // -- THE DIFFERENTIAL, against real fleet CSVs -------------------------
    if (argc > 1) {
        DIR* d = opendir(argv[1]);
        if (d == nullptr) {
            std::printf("  FAIL  cannot open dir %s\n", argv[1]);
            ++g_failures;
        } else {
            int files = 0, rows = 0;
            struct dirent* de = nullptr;
            while ((de = readdir(d)) != nullptr) {
                const std::string name(de->d_name);
                if (name.size() < 4 || name.compare(name.size() - 4, 4, ".csv") != 0) {
                    continue;
                }
                const std::string path = std::string(argv[1]) + "/" + name;
                std::ifstream f(path, std::ios::binary);
                if (!f.is_open()) continue;
                const std::string csv((std::istreambuf_iterator<char>(f)),
                                      std::istreambuf_iterator<char>());

                std::vector<KlineBar> mine, ref;
                ParseError err{};
                const bool okMine = parse(csv, mine, err);
                const bool okRef = refParse(csv, kBarSec, ref);
                check(okMine == okRef, name + ": both parsers agree on accept/reject");
                if (!okMine || !okRef) continue;
                check(mine.size() == ref.size(), name + ": same bar count");
                if (mine.size() != ref.size()) continue;
                bool same = true;
                for (size_t i = 0; i < mine.size(); ++i) {
                    if (!sameBar(mine[i], ref[i])) same = false;
                }
                check(same, name + ": every bar bit-identical to the old parser");
                ++files;
                rows += static_cast<int>(mine.size());
            }
            closedir(d);
            std::printf("  differential: %d file(s), %d row(s)\n", files, rows);
            // An empty directory passing silently would report "differential
            // clean" having compared nothing at all.
            check(files > 0, "the differential directory held at least one CSV");
        }
    } else {
        std::printf("  (no differential: pass a directory of real CSVs as argv[1])\n");
    }

    std::printf("\n=== %s: %d checks, %d failures ===\n",
                g_failures == 0 ? "BACKFILL CSV PASS" : "BACKFILL CSV FAIL",
                g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
