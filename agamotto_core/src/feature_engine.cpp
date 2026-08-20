#include "feature_engine.hpp"

#include "codes_generated.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

// ---------------------------------------------------------------------------
// FP CONTRACTION IS BANNED IN THIS TRANSLATION UNIT.
//
// The contract of this file is not "compute the correlation"; it is "perform
// the SAME SEQUENCE OF IEEE-754 double roundings numpy/pandas performs".
// numpy is a chain of separate ufuncs, so EVERY operation rounds to double.
// A compiler that contracts `a - b*c` into a single fused multiply-add keeps
// the product at infinite precision and rounds ONCE — a different function.
//
// That is not a cosmetic 1-ULP difference, because three predicates in here
// are decided by whether a cancellation lands EXACTLY on zero:
//
//   rollCorr   num == 0 over a zero denominator  ->  NaN;  num != 0  ->  +/-inf
//   rollSkew   B <= 1e-14                        ->  NaN
//   rollKurt   B <= 1e-14                        ->  NaN
//   rollStd    sqrt(var) with var < 0            ->  NaN   (pandas does not
//                                                           clamp ssqdm; see
//                                                           calc_var, no
//                                                           `if result < 0`)
//
// MEASURED, 2026-08-19, on tests/pdops_golden.py (seed 20260819):
// Apple clang 21 / arm64 defaults to -ffp-contract=on and arm64 has FMA in the
// base ISA, so `mXY[i] - mX[i]*mY[i]` became one fnmsub. On the 7 rows of
// `rollcorr|close|vol|14|14` where the vol window is constant at 1234.5, the
// separately-rounded product equals mXY EXACTLY (num = 0 -> NaN) while the
// fused form returns the true residual 3.9e-11 (num != 0 -> inf). gcc 8.5 on
// rockylinux:8 targets baseline x86-64, which has NO FMA instruction, so it
// could not contract and matched pandas by accident of the target ISA. The
// gate was therefore GREEN on Linux and RED on macOS from the same source.
//
// The DEFENCE IS IN THREE LAYERS, deliberately, because only the first of them
// is guaranteed by the language:
//
//   1. pdRound() below — a volatile round-trip at every site where the last
//      bit decides a NaN mask. Volatile accesses are observable behaviour, so
//      EVERY conforming compiler must materialise the value at type double.
//      This survives -ffp-contract=fast, -march=native, x87 excess precision
//      and an unknown future compiler. It is the layer that must not be
//      removed.
//   2. The pragma below — kills contraction for the whole TU on clang, so the
//      remaining (value-only, <=1 ULP) sites agree too. clang-only: GCC still
//      does not implement #pragma STDC FP_CONTRACT for C++ (GCC PR 20785) and
//      #pragma GCC optimize resets unrelated flags, so it is not used here.
//   3. -ffp-contract=off in CMakeLists on both the library and the test
//      targets, for compilers that honour neither pragma.
//
// Layers 2 and 3 are conveniences that keep the value diff at 0 ULP. Layer 1
// is the correctness guarantee: if 2 and 3 are ever dropped, the NaN masks
// still match and the 1e-12 value gate still passes.
// ---------------------------------------------------------------------------
// -ffast-math / -Ofast is NOT a "layer 4" this file can tolerate, and it is the
// one setting no barrier can rescue: it tells the compiler NaN and inf do not
// exist (clang literally warns "use of infinity is undefined behavior" on the
// NA constant below), which is the exact opposite of this file's contract —
// every primitive here MUST propagate pandas' NaN masks bit for bit. Measured
// 2026-08-19: -Ofast turns the 67-column gate into 15 failures. Refuse to
// build rather than ship a library whose warmup NaNs have been optimised away.
#if defined(__FAST_MATH__)
#  error "agamotto::pdops must not be built with -ffast-math/-Ofast: it assumes \
NaN/inf do not exist, and every pdops primitive reproduces pandas' NaN masks."
#endif

#if defined(__clang__)
#  pragma clang fp contract(off)
#endif

namespace agamotto {

namespace {
constexpr double NA = std::numeric_limits<double>::quiet_NaN();
inline bool isna(double v) { return std::isnan(v); }

// Force `v` to be rounded to double HERE, before it is used again.
//
// C++ has no operator meaning "round at this point" — contraction, x87 excess
// precision and reassociation are all free to carry an intermediate wider than
// double. A volatile store/load is the one construct the standard requires to
// be materialised at the declared type, so it is the only portable barrier.
//
// Cost is one L1 store + load. It is applied ONLY at the handful of sites
// where a rounding difference changes a NaN mask (see the banner above), never
// blanket-applied to the accumulator loops, so the hot path keeps its
// vectorisation.
inline double pdRound(double v)
{
    volatile double t = v;
    return t;
}
} // namespace

void Table::put(const std::string& n, std::vector<double> v)
{
    auto it = idx.find(n);
    if (it == idx.end()) {
        idx[n] = names.size();
        names.push_back(n);
        cols.push_back(std::move(v));
    } else {
        cols[it->second] = std::move(v);
    }
}

const std::vector<double>& Table::get(const std::string& n) const
{
    auto it = idx.find(n);
    if (it == idx.end()) throw std::out_of_range("Table::get: no column " + n);
    return cols[it->second];
}

std::vector<double>& Table::mut(const std::string& n)
{
    auto it = idx.find(n);
    if (it == idx.end()) throw std::out_of_range("Table::mut: no column " + n);
    return cols[it->second];
}

namespace pdops {

namespace {

// pandas FixedWindowIndexer.get_window_bounds for a right-aligned window:
//   end[i]   = i + 1
//   start[i] = clip(i + 1 - w, 0, n)
inline size_t winStart(size_t i, int w) { return (i + 1 >= static_cast<size_t>(w)) ? i + 1 - w : 0; }

// Kahan-compensated running accumulate, byte-for-byte the shape used by
// aggregations.pyx add_sum/add_mean/add_skew/add_kurt. The compensation is
// what keeps a streaming sum from drifting away from pandas' own; dropping it
// costs ~1e-11 on a 700-bar window of prices.
inline void kahanAdd(double& acc, double& comp, double val)
{
    const double y = val - comp;
    const double t = acc + y;
    comp = t - acc - y;
    acc = t;
}

// pandas Rolling._prep_values (rolling.py:375-378) runs on EVERY rolling call:
//
//     inf = np.isinf(values)
//     if inf.any(): values = np.where(inf, np.nan, values)
//
// So +/-inf is an ABSENT observation to every rolling statistic, not a huge
// one. This matters as soon as stage 2.2 lands: pct_change on a zero
// denominator yields inf (research.py:475 divides by a rolling volume mean),
// and an inf that reached a Kahan accumulator would poison every subsequent
// window of that series — the whole column downstream of one flat bar.
std::vector<double> prepValues(const std::vector<double>& x)
{
    std::vector<double> o(x);
    for (double& v : o)
        if (std::isinf(v)) v = NA;
    return o;
}

void requireWindow(const char* who, int w, int mp)
{
    // A non-positive window or a negative min_periods is a caller bug, not a
    // value to normalise away: pandas raises, and a silent clamp here would
    // turn "I passed the wrong variable" into a plausible column of numbers.
    if (w <= 0) throw std::invalid_argument(std::string(who) + ": window must be > 0");
    if (mp < 0) throw std::invalid_argument(std::string(who) + ": min_periods must be >= 0");
}

} // namespace

std::vector<double> diff(const std::vector<double>& x) { return diffN(x, 1); }

std::vector<double> diffN(const std::vector<double>& x, int n)
{
    if (n <= 0) throw std::invalid_argument("diffN: n must be > 0");
    std::vector<double> o(x.size(), NA);
    for (size_t i = static_cast<size_t>(n); i < x.size(); ++i) o[i] = x[i] - x[i - n];
    return o;
}

std::vector<double> shift(const std::vector<double>& x, int n)
{
    // Only the forward shift the reference uses (research.py:385-387, :478-480).
    // A negative shift is lookahead; refuse it rather than quietly obliging.
    if (n <= 0) throw std::invalid_argument("shift: n must be > 0 (a negative shift is lookahead)");
    std::vector<double> o(x.size(), NA);
    for (size_t i = static_cast<size_t>(n); i < x.size(); ++i) o[i] = x[i - n];
    return o;
}

std::vector<double> pctChange(const std::vector<double>& x, int periods)
{
    if (periods <= 0) throw std::invalid_argument("pctChange: periods must be > 0");
    std::vector<double> o(x.size(), NA);
    for (size_t i = static_cast<size_t>(periods); i < x.size(); ++i) {
        const double prev = x[i - periods];
        if (isna(x[i]) || isna(prev)) continue;
        // pandas yields +/-inf on a zero denominator (and NaN for 0/0) rather
        // than raising. Preserve it: the sanitisation pass that turns inf into
        // 0.0 belongs to the feature layer, not here, and swallowing it would
        // hide a genuinely zero price/volume.
        o[i] = x[i] / prev - 1.0;
    }
    return o;
}

// ---------------------------------------------------------------------------
// rolling sum / mean
// ---------------------------------------------------------------------------
namespace {

// Shared streaming driver for the sum/mean pair: both maintain a Kahan sum
// plus the observation count, the count of NEGATIVE observations (pandas'
// calc_mean clamps a non-negative window whose mean rounds below zero), and
// the trailing run of equal values (pandas' GH#42064 guard, which forces an
// all-equal window to prev*nobs instead of the accumulated residue).
struct SumState {
    double sum = 0.0, compAdd = 0.0, compRem = 0.0, prev = 0.0;
    long long nobs = 0, neg = 0, sameRun = 0;

    void reset(double firstInWindow)
    {
        sum = compAdd = compRem = 0.0;
        nobs = neg = sameRun = 0;
        prev = firstInWindow;
    }
    void add(double v)
    {
        if (isna(v)) return;
        ++nobs;
        kahanAdd(sum, compAdd, v);
        if (std::signbit(v)) ++neg;
        sameRun = (v == prev) ? sameRun + 1 : 1;
        prev = v;
    }
    void remove(double v)
    {
        if (isna(v)) return;
        --nobs;
        kahanAdd(sum, compRem, -v);
        if (std::signbit(v)) --neg;
    }
};

template <typename Calc>
std::vector<double> streamSum(const std::vector<double>& x, int w, int mp, Calc calc)
{
    const size_t n = x.size();
    std::vector<double> o(n, NA);
    SumState st;
    for (size_t i = 0; i < n; ++i) {
        const size_t s = winStart(i, w), e = i + 1;
        if (i == 0 || s >= i) {           // s >= end[i-1] == i  -> window restarted
            st.reset(x[s]);
            for (size_t j = s; j < e; ++j) st.add(x[j]);
        } else {
            for (size_t j = winStart(i - 1, w); j < s; ++j) st.remove(x[j]);
            st.add(x[i]);
        }
        o[i] = calc(st, mp);
    }
    return o;
}

} // namespace

// The *Raw variants skip prepValues, because rollCorr must sanitise ONCE,
// before prep_binary's product and count are formed — exactly the order
// rolling.py::cov/corr_func uses. The public entry points below sanitise.
static std::vector<double> rollSumRaw(const std::vector<double>& x, int w, int mp)
{
    requireWindow("rollSum", w, mp);
    return streamSum(x, w, mp, [](const SumState& st, int minp) {
        if (st.nobs == 0 && minp == 0) return 0.0;
        if (st.nobs < minp) return NA;
        return (st.sameRun >= st.nobs) ? st.prev * static_cast<double>(st.nobs) : st.sum;
    });
}

static std::vector<double> rollMeanRaw(const std::vector<double>& x, int w, int mp)
{
    requireWindow("rollMean", w, mp);
    // roll_mean does NOT clamp minp (unlike roll_var/roll_skew/roll_kurt);
    // calc_mean's own `nobs > 0` is what rejects an empty window.
    return streamSum(x, w, mp, [](const SumState& st, int m) {
        if (st.nobs < m || st.nobs == 0) return NA;
        // calc_mean, exactly: the same-value guard REPLACES the mean with the
        // repeated value itself (NOT prev*nobs/nobs, which re-introduces the
        // rounding it exists to remove), and it SHORT-CIRCUITS the sign clamps.
        double r = st.sum / static_cast<double>(st.nobs);
        if (st.sameRun >= st.nobs) r = st.prev;
        else if (st.neg == 0 && r < 0.0) r = 0.0;         // all non-negative
        else if (st.neg == st.nobs && r > 0.0) r = 0.0;   // all negative
        return r;
    });
}

std::vector<double> rollSum(const std::vector<double>& x, int w, int mp)
{
    return rollSumRaw(prepValues(x), w, mp);
}

std::vector<double> rollMean(const std::vector<double>& x, int w, int mp)
{
    return rollMeanRaw(prepValues(x), w, mp);
}

// ---------------------------------------------------------------------------
// rolling var / std  — Welford add/remove with Kahan compensation
// ---------------------------------------------------------------------------
static std::vector<double> rollVarRaw(const std::vector<double>& x, int w, int mp, int ddof)
{
    requireWindow("rollVar", w, mp);
    if (ddof < 0) throw std::invalid_argument("rollVar: ddof must be >= 0");
    const size_t n = x.size();
    const int minp = std::max(mp, 1);
    std::vector<double> o(n, NA);

    double nobs = 0.0, mean = 0.0, ssqdm = 0.0, compAdd = 0.0, compRem = 0.0, prev = 0.0;
    long long sameRun = 0;

    auto add = [&](double v) {
        if (isna(v)) return;
        nobs += 1.0;
        sameRun = (v == prev) ? sameRun + 1 : 1;
        prev = v;
        const double prevMean = mean - compAdd;
        const double y = v - compAdd;
        const double t = y - mean;
        compAdd = t + mean - y;
        mean += t / nobs;
        // pdRound here buys DETERMINISM ACROSS OUR TOOLCHAINS, not pandas
        // parity — and the distinction matters, so read before "simplifying".
        //
        // add_var is CYTHON, compiled to C, so pandas' own last bit at this
        // site is whatever the C compiler that built the wheel decided. It is
        // NOT stable: measured 2026-08-19, the pandas 2.3.3 arm64 wheel DOES
        // contract this into an FMA — a bit-exact Python replica of add_var
        // matches its roll_var on 0 of 1965 rows with separate rounding and on
        // 1965 of 1965 with math.fma — while a baseline-x86-64 build cannot
        // contract at all. So no choice here is "the pandas answer"; there
        // are two, and which one a golden carries depends on the machine that
        // generated it.
        //
        // Given that, this takes the form the reference SOURCE is written in
        // (a separately-rounded product) rather than encoding one wheel's
        // build flags into the library, and pins it so clang and gcc cannot
        // disagree with EACH OTHER — which is the property the live core needs,
        // since it is built on dev105 and developed on macOS. The residual
        // 1-ULP gap to any given wheel is what the 1e-12 value gate is for;
        // the specs where conditioning amplifies it past that are PROBE.
        //
        // The NaN masks do NOT depend on this choice: rollStd goes through
        // pandas' zsqrt (negative -> 0.0, see rollStd), and the SIGN of a
        // negative ssqdm residue is robust — verified, 0 sign changes in 1965
        // rows between the fused and unfused forms.
        ssqdm += pdRound((v - prevMean) * (v - mean));
    };
    auto remove = [&](double v) {
        if (isna(v)) return;
        nobs -= 1.0;
        if (nobs > 0.0) {
            const double prevMean = mean - compRem;
            const double y = v - compRem;
            const double t = y - mean;
            compRem = t + mean - y;
            mean -= t / nobs;
            ssqdm -= pdRound((v - prevMean) * (v - mean));   // see add(), above
        } else {
            mean = 0.0;
            ssqdm = 0.0;
        }
    };

    for (size_t i = 0; i < n; ++i) {
        const size_t s = winStart(i, w), e = i + 1;
        if (i == 0 || s >= i) {
            nobs = mean = ssqdm = compAdd = compRem = 0.0;
            sameRun = 0;
            prev = x[s];
            for (size_t j = s; j < e; ++j) add(x[j]);
        } else {
            for (size_t j = winStart(i - 1, w); j < s; ++j) remove(x[j]);
            add(x[i]);
        }
        if (nobs >= static_cast<double>(minp) && nobs > static_cast<double>(ddof)) {
            // All-equal window: pandas returns EXACTLY 0, not the residue of
            // the running update. Without this the value is ~1e-17 and any
            // ratio built on it (corr's denominator) flips sign at random.
            o[i] = (nobs == 1.0 || static_cast<double>(sameRun) >= nobs)
                       ? 0.0
                       : ssqdm / (nobs - static_cast<double>(ddof));
        }
    }
    return o;
}

std::vector<double> rollVar(const std::vector<double>& x, int w, int mp, int ddof)
{
    return rollVarRaw(prepValues(x), w, mp, ddof);
}

std::vector<double> rollStd(const std::vector<double>& x, int w, int mp)
{
    std::vector<double> v = rollVar(x, w, mp, 1);
    // pandas' Rolling.std is zsqrt(var), NOT sqrt(var) — pandas/core/window/
    // rolling.py:1460-1462 -> common.py:149-161:
    //
    //     result = np.sqrt(x); mask = x < 0; result[mask] = 0
    //
    // A NEGATIVE variance is not hypothetical: calc_var has no clamp, and the
    // streaming Welford add/remove accumulates a cancellation residue that
    // goes negative whenever a window is near-constant at a value far from
    // zero. Measured on tests/pdops_golden.py's `flat` series (blocks of
    // 1.0 / 1.0000001, true variance 2.5e-15): pandas' rolling var returns
    // -3.1e-10 on 170 of 2000 rows, and pandas' std returns 0.0 on every one
    // of them. A bare std::sqrt returns NaN there — 170 rows of NaN-mask
    // divergence, the same failure class as the FMA defect.
    //
    // This IS a clamp on a derived quantity, which CLAUDE.md bans by default.
    // It is deliberate and it is not hiding a bug of ours: reproducing pandas
    // bit for bit is this file's entire contract, and `x < 0 -> 0` is the
    // reference's own documented behaviour. Removing it does not surface a
    // defect, it manufactures one. NaN is preserved because `NaN < 0` is
    // false, exactly as numpy's mask behaves.
    for (double& e : v) e = (e < 0.0) ? 0.0 : std::sqrt(e);
    return v;
}

// ---------------------------------------------------------------------------
// rolling skew / kurt  — streaming RAW moments, pandas' calc_skew/calc_kurt
// ---------------------------------------------------------------------------
namespace {

// pandas roll_skew / roll_kurt pre-centre the WHOLE array before accumulating
// raw moments: `mean_val = round(nanmean(values))` is subtracted from a copy,
// but only when `nanmin(values) - mean_val > guard` (guard = -1e5 for skew,
// -1e4 for kurt — they really do differ).
//
// This is NOT cosmetic and it is NOT window-local. Raw moments of a price-scale
// series cancel catastrophically (sum(x^4) ~ 1e19 against a 4th central moment
// of ~1e7), and the centring is what keeps the answer meaningful. Reproducing
// it is also what makes this port agree with pandas to 1e-13 instead of 1e-4 on
// such a series — measured 2026-08-19, tests/pdops_parity_driver PROBE rows.
//
// The consequence to keep in mind: the constant depends on the WHOLE panel, so
// pandas' rolling skew of a given window changes when the frame start moves
// (pdops_golden.py prints that self-disagreement). Nothing here can fix that;
// it is a property of the reference.
std::vector<double> preCentre(const std::vector<double>& x, double guard)
{
    long long nobsMean = 0;
    double sumVal = 0.0;
    double minVal = std::numeric_limits<double>::infinity();
    for (double v : x) {
        if (isna(v)) continue;
        ++nobsMean;
        sumVal += v;
        if (v < minVal) minVal = v;
    }
    if (nobsMean == 0) return x;   // nanmin/nanmean are NaN; the guard is False
    const double meanVal = std::round(sumVal / static_cast<double>(nobsMean));
    if (!(minVal - (sumVal / static_cast<double>(nobsMean)) > guard)) return x;
    std::vector<double> out(x.size());
    for (size_t i = 0; i < x.size(); ++i) out[i] = x[i] - meanVal;
    return out;
}

// order == 3 for skew, 4 for kurt. m[k] is the running sum of val^k over the
// CENTRED copy; `raw` is the original, because pandas seeds prev_value from
// `values[s]` (uncentred) while feeding add_skew from `values_copy` (centred).
template <int ORDER, typename Calc>
std::vector<double> streamMoments(const std::vector<double>& raw, const std::vector<double>& x,
                                  int w, int mp, Calc calc)
{
    const size_t n = x.size();
    std::vector<double> o(n, NA);
    double m[ORDER + 1] = {0.0};
    double cAdd[ORDER + 1] = {0.0};
    double cRem[ORDER + 1] = {0.0};
    long long nobs = 0, sameRun = 0;
    double prev = 0.0;

    auto add = [&](double v) {
        if (isna(v)) return;
        ++nobs;
        double p = 1.0;
        for (int k = 1; k <= ORDER; ++k) { p *= v; kahanAdd(m[k], cAdd[k], p); }
        sameRun = (v == prev) ? sameRun + 1 : 1;
        prev = v;
    };
    auto remove = [&](double v) {
        if (isna(v)) return;
        --nobs;
        double p = 1.0;
        for (int k = 1; k <= ORDER; ++k) { p *= v; kahanAdd(m[k], cRem[k], -p); }
    };

    for (size_t i = 0; i < n; ++i) {
        const size_t s = winStart(i, w), e = i + 1;
        if (i == 0 || s >= i) {
            for (int k = 0; k <= ORDER; ++k) m[k] = cAdd[k] = cRem[k] = 0.0;
            nobs = sameRun = 0;
            // Deliberately the UNCENTRED value — pandas' roll_skew/roll_kurt
            // seed prev_value from `values[s]`, not `values_copy[s]`. It only
            // matters at a window restart, and only for the same-value guard,
            // but reproducing it costs nothing and guessing costs parity.
            prev = raw[s];
            for (size_t j = s; j < e; ++j) add(x[j]);
        } else {
            for (size_t j = winStart(i - 1, w); j < s; ++j) remove(x[j]);
            add(x[i]);
        }
        o[i] = calc(m, nobs, sameRun, mp);
    }
    return o;
}

} // namespace

std::vector<double> rollSkew(const std::vector<double>& x, int w, int mp)
{
    requireWindow("rollSkew", w, mp);
    const int minp = std::max(mp, 3);   // pandas: minp = max(minp, 3)
    const std::vector<double> v = prepValues(x);
    const std::vector<double> c = preCentre(v, -1e5);   // roll_skew's guard
    return streamMoments<3>(v, c, w, minp, [](const double* m, long long nobs,
                                              long long sameRun, int p) {
        if (nobs < p || nobs < 3) return NA;
        // Every observation identical: pandas forces 0.0 rather than the
        // garbage that dividing a cancelled C by a cancelled R^3 produces.
        if (sameRun >= nobs) return 0.0;
        const double dn = static_cast<double>(nobs);
        const double A = m[1] / dn;
        // B is compared against a hard 1e-14 to decide NaN, so it is a MASK
        // predicate, and `m[2]/dn - A*A` is the canonical FMA pattern.
        // pdRound pins it so clang and gcc agree with each other.
        //
        // It does NOT make the guard reproducible against pandas, and nothing
        // can: on a near-constant window B is pure cancellation residue, and
        // pandas disagrees with ITSELF about it — moving only the frame start
        // flips 101-113 rows of the skew/kurt NaN mask (measured on the
        // `nearflat` series). That is why no near-constant skew/kurt spec is
        // gated anywhere in this suite; see tests/pdops_golden.py.
        const double B = pdRound(m[2] / dn - A * A);
        const double C = m[3] / dn - A * A * A - 3.0 * A * B;
        // pandas' #18044 guard: below this the variance is floating-point
        // noise and the ratio explodes.
        if (B <= 1e-14) return NA;
        const double R = std::sqrt(B);
        // SAMPLE (bias-corrected) G1. sqrt(n(n-1))*C/((n-2)R^3) is algebraically
        // n/((n-1)(n-2)) * M3 / s^3 with s the ddof=1 std; the POPULATION form
        // C/R^3 is 12.4% smaller at n=14 (tests/pdops_golden.py NEG_popskew
        // asserts we do not compute it).
        return (std::sqrt(dn * (dn - 1.0)) * C) / ((dn - 2.0) * R * R * R);
    });
}

std::vector<double> rollKurt(const std::vector<double>& x, int w, int mp)
{
    requireWindow("rollKurt", w, mp);
    const int minp = std::max(mp, 4);   // pandas: minp = max(minp, 4)
    // -1e4, NOT -1e5: roll_kurt and roll_skew genuinely use different guards.
    const std::vector<double> v = prepValues(x);
    const std::vector<double> c = preCentre(v, -1e4);
    return streamMoments<4>(v, c, w, minp, [](const double* m, long long nobs,
                                              long long sameRun, int p) {
        if (nobs < p || nobs < 4) return NA;
        if (sameRun >= nobs) return -3.0;   // excess kurtosis of a constant
        const double dn = static_cast<double>(nobs);
        const double A = m[1] / dn;
        double R = A * A;
        const double B = pdRound(m[2] / dn - R);   // MASK predicate; see rollSkew
        R = R * A;
        const double C = m[3] / dn - R - 3.0 * A * B;
        R = R * A;
        const double D = m[4] / dn - R - 6.0 * B * A * A - 4.0 * C * A;
        if (B <= 1e-14) return NA;
        // SAMPLE EXCESS G2. The population form D/B^2 - 3 is ~40% off at n=14
        // (tests/pdops_golden.py NEG_popkurt asserts we do not compute it).
        const double K = (dn * dn - 1.0) * D / (B * B) - 3.0 * ((dn - 1.0) * (dn - 1.0));
        return K / ((dn - 2.0) * (dn - 3.0));
    });
}

// ---------------------------------------------------------------------------
// rolling corr
// ---------------------------------------------------------------------------
std::vector<double> rollCorr(const std::vector<double>& x, const std::vector<double>& y,
                             int w, int mp)
{
    requireWindow("rollCorr", w, mp);
    if (x.size() != y.size())
        throw std::invalid_argument("rollCorr: series lengths differ");
    const size_t n = x.size();

    // pandas/core/window/rolling.py prep_binary: X = x + 0*y, Y = y + 0*x.
    // BOTH series are masked wherever EITHER is NaN, BEFORE any rolling stat.
    // This is the single easiest thing to get wrong here: computing each mean
    // and variance over that series' OWN non-NaN values gives a plausible
    // number that is simply a different statistic. Measured on the golden,
    // the un-masked form differs by 100%+ on the rows after a NaN run
    // (research.py:589 correlates hist_return with hist_return.shift(1), so
    // the two masks differ by exactly one row at every hole).
    // Written as pandas writes it, not as an equivalent-looking mask: `0*y` is
    // NaN when y is NaN AND when y is +/-inf, so the two are not the same
    // predicate. prep_binary runs on the RAW series; _prep_values (inf -> NaN)
    // runs after, inside corr_func — that order is preserved here.
    std::vector<double> X(n), Y(n);
    for (size_t i = 0; i < n; ++i) {
        X[i] = x[i] + 0.0 * y[i];
        Y[i] = y[i] + 0.0 * x[i];
    }
    X = prepValues(X);
    Y = prepValues(Y);

    std::vector<double> XY(n), notna(n);
    for (size_t i = 0; i < n; ++i) {
        XY[i] = X[i] * Y[i];
        notna[i] = isna(X[i] + Y[i]) ? 0.0 : 1.0;
    }

    // *Raw: X, Y and XY are already sanitised, and pandas calls the window
    // kernels DIRECTLY here (not through _apply), so there is no second pass.
    const std::vector<double> mXY = rollMeanRaw(XY, w, mp);
    const std::vector<double> mX = rollMeanRaw(X, w, mp);
    const std::vector<double> mY = rollMeanRaw(Y, w, mp);
    const std::vector<double> cnt = rollSumRaw(notna, w, 0);
    const std::vector<double> vX = rollVarRaw(X, w, mp, 1);
    const std::vector<double> vY = rollVarRaw(Y, w, mp, 1);

    std::vector<double> o(n, NA);
    for (size_t i = 0; i < n; ++i) {
        // THE mask predicate of this whole file. When either window is
        // constant its variance is EXACTLY 0 (calc_var's same-value guard), so
        // the denominator is 0 and the result is decided entirely by whether
        // the numerator cancelled to exactly zero:
        //     num == 0  ->  0/0    ->  NaN
        //     num != 0  ->  x/0    ->  +/-inf
        // pandas produces BOTH on this golden — NaN on rows 514/517/518 of
        // rollcorr|close|vol|14|14 and +/-inf (num = +/-1 ULP of 8e7) on rows
        // 513/515/516/519 — so there is no "zero-variance -> NaN" shortcut
        // that is faithful. The only correct implementation is pandas' own
        // arithmetic with pandas' own roundings, which is what pdRound pins.
        const double prod = pdRound(mX[i] * mY[i]);
        const double num  = pdRound(mXY[i] - prod) * (cnt[i] / (cnt[i] - 1.0));
        o[i] = num / std::sqrt(vX[i] * vY[i]);
    }
    return o;
}

// ---------------------------------------------------------------------------
// rolling quantile
// ---------------------------------------------------------------------------
namespace {

// pandas roll_quantile, interpolation='linear', over the SORTED NON-NaN values.
// `nobs` is the NON-NaN COUNT, not the window width — the difference is the
// whole warmup behaviour of the high_vol_q* cutoffs.
inline double quantileOf(const std::vector<double>& sorted, double q)
{
    const size_t k = sorted.size();
    if (k == 1) return sorted[0];
    const double h = q * static_cast<double>(k - 1);
    const int lo = static_cast<int>(h);
    // pandas takes the exact-index shortcut BEFORE touching lo+1, which is
    // what keeps q=1.0 from reading off the end.
    if (h == static_cast<double>(lo)) return sorted[static_cast<size_t>(lo)];
    const double vlow = sorted[static_cast<size_t>(lo)];
    const double vhigh = sorted[static_cast<size_t>(lo) + 1];
    return vlow + (vhigh - vlow) * (h - static_cast<double>(lo));
}

} // namespace

std::vector<double> rollQuantile(const std::vector<double>& x, int w, int mp, double q)
{
    return rollQuantiles(x, w, mp, {q})[0];
}

std::vector<std::vector<double>> rollQuantiles(const std::vector<double>& x, int w, int mp,
                                               const std::vector<double>& qs)
{
    requireWindow("rollQuantiles", w, mp);
    if (qs.empty()) throw std::invalid_argument("rollQuantiles: no levels given");
    for (double q : qs)
        if (!(q >= 0.0 && q <= 1.0))
            throw std::invalid_argument("rollQuantiles: level outside [0, 1]");

    const std::vector<double> v = prepValues(x);
    const size_t n = v.size();
    const int minp = std::max(mp, 1);
    std::vector<std::vector<double>> out(qs.size(), std::vector<double>(n, NA));
    std::vector<double> buf;
    buf.reserve(static_cast<size_t>(w));

    for (size_t i = 0; i < n; ++i) {
        const size_t s = winStart(i, w);
        buf.clear();
        for (size_t j = s; j <= i; ++j)
            if (!isna(v[j])) buf.push_back(v[j]);
        if (buf.size() < static_cast<size_t>(minp) || buf.empty()) continue;
        // ONE sort serves every level — that is the only reason this function
        // exists next to rollQuantile, and the driver asserts the results are
        // cell-for-cell identical to the single-level calls.
        std::sort(buf.begin(), buf.end());
        for (size_t j = 0; j < qs.size(); ++j) out[j][i] = quantileOf(buf, qs[j]);
    }
    return out;
}

} // namespace pdops

// ===========================================================================
// STAGE 2.2 — the OHLC / returns / MA / volume feature blocks
// ===========================================================================
namespace {

// Elementwise a - b. A named helper rather than an inline loop at each site so
// every subtraction below is provably ONE rounded double operation, which is
// what numpy does (each ufunc rounds to double).
std::vector<double> subVec(const std::vector<double>& a, const std::vector<double>& b)
{
    std::vector<double> out(a.size());
    for (size_t i = 0; i < a.size(); ++i) out[i] = a[i] - b[i];
    return out;
}

// Elementwise a / (b + eps).
//
// `eps` is a PARAMETER passed as a LITERAL at every call site, deliberately.
// research.py writes `+ 1e-8` inline in each of the eight expressions listed in
// the header banner, and there is no shared EPS constant in this file to hoist
// it into — one named constant is a single edit away from silently re-scaling
// every ratio feature, and on a ~0.0045-priced symbol that edit moves the sixth
// significant figure of a top-5 IC-selected feature. Keeping the literal at the
// call site is what makes a line-by-line diff against research.py possible.
//
// Division by exactly zero yields +/-inf, exactly as pandas does. Nothing here
// clamps or sanitises it (see the header banner).
std::vector<double> ratioEps(const std::vector<double>& a, const std::vector<double>& b,
                             double eps)
{
    std::vector<double> out(a.size());
    for (size_t i = 0; i < a.size(); ++i) out[i] = a[i] / (b[i] + eps);
    return out;
}

void requireColumn(const std::vector<double>& v, const char* name, size_t n)
{
    if (v.size() != n)
        throw std::invalid_argument(
            std::string("engineerFeatures: column '") + name + "' has " +
            std::to_string(v.size()) + " rows, expected " + std::to_string(n));
}

// An OPTIONAL column is either absent (empty) or full width. A SHORT one is a
// caller bug, never a shorter history — fail rather than pad.
bool optionalPresent(const std::vector<double>& v, const char* name, size_t n)
{
    if (v.empty()) return false;
    requireColumn(v, name, n);
    return true;
}

// features_scalefree.py:61-63 `_safe`, and it is TWO OPERATIONS, not one:
//
//     out = num / den.replace(0.0, np.nan)          # step 1
//     return out.replace([np.inf, -np.inf], np.nan) # step 2
//
// STEP 1 removes an EXACTLY-zero denominator BEFORE the division, so the
// division never has the chance to produce the +/-inf that a zero denominator
// would give. pandas' `Series.replace(0.0, nan)` matches with `==`, and
// `-0.0 == 0.0` is true, so NEGATIVE ZERO is replaced as well — verified
// against pandas 2.3.3 (`pd.Series([-0.0]).replace(0.0, np.nan)` -> NaN). A
// `std::signbit` check or a `memcmp` against +0.0 would MISS -0.0 and leave a
// -inf where the reference has NaN. `d == 0.0` is the whole test.
//
// STEP 2 is a DIFFERENT case and is not implied by step 1: it catches an
// infinite NUMERATOR (inf / finite -> inf) and an overflow (a large finite
// numerator over a tiny-but-nonzero denominator). Implementing step 1 alone —
// the natural `if (den == 0.0) return NaN; return num / den;` — is therefore
// NOT equivalent, and tests/run_feature_parity.sh --negative-safeinf builds
// exactly that mutant and requires the gate to go red.
//
// Why this matters beyond parity: the docstring on `_safe` says it, and it is
// not decoration — an inf reaching
// `rolling_predict_returns._compute_train_window_top_n_ic` sorts to the top of
// the |IC| ranking and silently displaces a real feature from the model.
//
// NaN in either operand propagates through both steps untouched, which is the
// pandas behaviour: NaN/x, x/NaN and NaN/NaN are all NaN, and `replace` does
// not match NaN by equality.
std::vector<double> safeDiv(const std::vector<double>& num, const std::vector<double>& den)
{
    std::vector<double> out(num.size());
    for (size_t i = 0; i < num.size(); ++i) {
        const double d = (den[i] == 0.0) ? NA : den[i];  // step 1 (catches -0.0)
        const double r = num[i] / d;
        out[i] = std::isinf(r) ? NA : r;                 // step 2
    }
    return out;
}

// x.fillna(0.0) — EVERY NaN, not a head. research.py:589 applies it to
// acf_lag1 AFTER the rolling correlation, so a window rejected by min_periods
// and a window pandas' constant-guard turned into NaN both read 0.0.
// +/-inf is NOT touched: `fillna` fills NaN only. (The TA-Lib block has its own
// copy for obv/ad; they are in different translation units and neither is worth
// a shared header for four lines.)
std::vector<double> fillNaZeroLocal(std::vector<double> x)
{
    for (double& v : x)
        if (isna(v)) v = 0.0;
    return x;
}

} // namespace

Table engineerFeatures(const RawBars& bars)
{
    const size_t n = bars.close.size();
    if (n != PANEL_BARS)
        throw std::invalid_argument(
            "engineerFeatures: panel is " + std::to_string(n) + " bars, but live "
            "engineers exactly " + std::to_string(PANEL_BARS) + " closed bars "
            "(trading.py:443 limit=700 -> :480 tail -> :485 iloc[:-1]). Column " +
            std::string(codes::F_PRICE_RANGE_PCT_Q50) + " is an EXPANDING median "
            "below 700 rows, so a different width computes different numbers, "
            "not merely more of them.");
    // ^ THE COLUMN IS NAMED BY ITS CODE HERE, NOT BY ITS REAL NAME. A throw
    // message is a string literal and lands in the .so verbatim: this one
    // carried the ONLY distinctive name `strings libagamotto_core.so` could
    // recover (found 2026-08-20, by the artifact audit build_linux.sh now
    // runs). Exception text is the easiest place for a name to re-enter the
    // artifact, precisely because it does not look like data — keep every
    // column reference in this file symbolic.

    requireColumn(bars.open, "open", n);
    requireColumn(bars.high, "high", n);
    requireColumn(bars.low, "low", n);
    requireColumn(bars.volume, "volume", n);

    Table t;

    // --- passthrough -------------------------------------------------------
    // `close` is the one raw level the engineered panel carries forward
    // (research.py:594 heads the concat list with it) — the regime predicates
    // compare against it directly (`close > sar`, `close < bb_lower`).
    // Uncoded: a universal market-data field name, not strategy IP.
    t.put("close", bars.close);

    // --- OHLC block, research.py:361-380 -----------------------------------
    const std::vector<double> price_range = subVec(bars.high, bars.low);
    t.put(codes::F_PRICE_RANGE, price_range);

    // research.py:362 recomputes (high - low) rather than reusing price_range.
    // Identical doubles either way; written the same way so the two files diff.
    const std::vector<double> price_range_pct =
        ratioEps(subVec(bars.high, bars.low), bars.open, 1e-8);
    t.put(codes::F_PRICE_RANGE_PCT, price_range_pct);

    // research.py:363. min_periods=1 -> EXPANDING below 700 rows, which is what
    // makes PANEL_BARS a correctness parameter. The literal 700 is what the
    // reference writes at this site (not VOL_Q_WINDOW); same value.
    t.put(codes::F_PRICE_RANGE_PCT_Q50,
          pdops::rollQuantile(price_range_pct, 700, 1, 0.5));

    // research.py:371-376. min_periods=VOL_Q_WINDOW, so these are ALL-NaN on a
    // PANEL_BARS-wide panel. See the VOL_Q_WINDOW banner in the header: live
    // behaviour under an open production finding, reproduced and NOT fixed.
    // One sort per window serves all three levels; the pdops driver asserts
    // rollQuantiles is cell-identical to three rollQuantile calls.
    {
        const std::vector<std::vector<double>> q =
            pdops::rollQuantiles(price_range_pct, VOL_Q_WINDOW, VOL_Q_WINDOW,
                                 {0.80, 0.90, 0.95});
        t.put(codes::F_PRICE_RANGE_PCT_Q80, q[0]);
        t.put(codes::F_PRICE_RANGE_PCT_Q90, q[1]);
        t.put(codes::F_PRICE_RANGE_PCT_Q95, q[2]);
    }

    const std::vector<double> open_close_diff = subVec(bars.close, bars.open);
    t.put(codes::F_OPEN_CLOSE_DIFF, open_close_diff);
    t.put(codes::F_OPEN_CLOSE_PCT, ratioEps(open_close_diff, bars.open, 1e-8));
    t.put(codes::F_HIGH_OPEN_PCT,
          ratioEps(subVec(bars.high, bars.open), bars.open, 1e-8));
    t.put(codes::F_LOW_OPEN_PCT,
          ratioEps(subVec(bars.low, bars.open), bars.open, 1e-8));

    // --- returns block, research.py:382-387 --------------------------------
    // hist_return = close.pct_change(fill_method=None); its LAGS are features.
    // The very next line of research.py is
    //   price_return = hist_return.shift(-1)
    // which is the TARGET, and is deliberately not computed anywhere here.
    const std::vector<double> hist_return = pdops::pctChange(bars.close, 1);
    t.put(codes::F_RET_LAG1, pdops::shift(hist_return, 1));
    t.put(codes::F_RET_LAG2, pdops::shift(hist_return, 2));
    t.put(codes::F_RET_LAG3, pdops::shift(hist_return, 3));

    // --- moving averages, research.py:461-468 ------------------------------
    // MA_PERIODS default [7, 25, 99], renamed to mvg1/mvg2/mvg3 on the way into
    // the wide frame (research.py:613-615) — that is the name the regime
    // predicates (MVG_DEPENDENT_FILTERS) and the vertical panel use.
    // dc/obfuscation/map.json has no code for them, so the real names stand.
    t.put("mvg1", pdops::rollMean(bars.close, 7, 1));
    t.put("mvg2", pdops::rollMean(bars.close, 25, 1));
    t.put("mvg3", pdops::rollMean(bars.close, 99, 1));

    // --- volume block, research.py:470-499 ---------------------------------
    // Guarded on column PRESENCE exactly as the reference is, and with the
    // reference's NESTING preserved: buy_pressure and trade_intensity sit
    // INSIDE the `quote_volume` branch (research.py:483-499), so a feed with
    // number_of_trades but no quote_volume emits no trade_intensity. That is
    // the reference's shape, quirk included — see README.
    {
        const std::vector<double> vol_ma = pdops::rollMean(bars.volume, 7, 1);
        t.put(codes::F_VOL_RATIO, ratioEps(bars.volume, vol_ma, 1e-8));
        const std::vector<double> vol_ret = pdops::pctChange(bars.volume, 1);
        t.put(codes::F_VOL_RET_LAG1, pdops::shift(vol_ret, 1));
        t.put(codes::F_VOL_RET_LAG2, pdops::shift(vol_ret, 2));
        t.put(codes::F_VOL_RET_LAG3, pdops::shift(vol_ret, 3));
    }

    if (optionalPresent(bars.quote_volume, "quote_volume", n)) {
        const std::vector<double> qv_ma = pdops::rollMean(bars.quote_volume, 7, 1);
        t.put(codes::F_QUOTE_VOL_RATIO, ratioEps(bars.quote_volume, qv_ma, 1e-8));

        if (optionalPresent(bars.taker_buy_quote_volume, "taker_buy_quote_volume", n))
            t.put(codes::F_BUY_PRESSURE,
                  ratioEps(bars.taker_buy_quote_volume, bars.quote_volume, 1e-8));

        if (optionalPresent(bars.number_of_trades, "number_of_trades", n)) {
            const std::vector<double> tr_ma = pdops::rollMean(bars.number_of_trades, 7, 1);
            t.put(codes::F_TRADE_INTENSITY,
                  ratioEps(bars.number_of_trades, tr_ma, 1e-8));
        }
    }

    // --- TA-Lib block, research.py:501-553 ---------------------------------
    // 25 ta-lib calls -> 29 columns, plus parkinson_vol, which is not a ta-lib
    // call but lives inside the reference's `try:` and so shares its fate.
    // Implemented in src/talib_block.cpp because that is the ONLY translation
    // unit in the core that needs a ta-lib include and a ta-lib link; keeping
    // it out of this file keeps the dependency visible in the build recipe.
    //
    // Appended LAST, matching the reference's concat order (`ta_features` goes
    // in after the volume features). Column ORDER is not what the gate compares
    // — it matches by name — but a diff-stable panel is worth having, and
    // matching the reference keeps the two files readable side by side.
    {
        std::vector<std::pair<std::string, std::vector<double>>> ta;
        talib_block::compute(bars.open, bars.high, bars.low, bars.close, bars.volume, ta);
        for (auto& kv : ta) t.put(kv.first, std::move(kv.second));
    }

    // --- rolling stats, research.py:557-593 --------------------------------
    // ALL FOUR TAKE `hist_return`, NOT `close`. research.py:569-571 and :587
    // each pass `hist_return` — the pct_change series computed above, not the
    // price series two lines further up. Getting this wrong is not a rounding
    // difference: on BTC the price-scale std is ~1e4 and the return-scale one
    // ~4e-3.
    //
    // *** THIS IS WHERE `f085` COMES FROM, AND THE ONLY PLACE. ***
    // sentinel_core emits the same code from TA_STDDEV(close, 14) — a
    // price-scale quantity. agamotto's is the return-scale one. Stage 2.3
    // deliberately left f085 out of talib_block.cpp so the two definitions
    // cannot coexist in this binary. See the header banner.
    //
    // min_periods is passed EXPLICITLY as the pandas default (== the window):
    // research.py writes `hist_return.rolling(window=stats_window)` with no
    // min_periods, and pdops has no defaults on purpose.
    t.put(codes::F_STD,  pdops::rollStd (hist_return, STATS_WINDOW, STATS_WINDOW));
    t.put(codes::F_SKEW, pdops::rollSkew(hist_return, STATS_WINDOW, STATS_WINDOW));
    t.put(codes::F_KURT, pdops::rollKurt(hist_return, STATS_WINDOW, STATS_WINDOW));

    // research.py:582-590. `pd.Series.autocorr(lag=1)` over a w-wide window IS
    // corr(x[:-1], x[1:]), so the reference expresses it as a (w-1)-WIDE rolling
    // correlation of the series against its own lag-1 shift — window 13, not 14.
    // The `.fillna(0.0)` runs AFTER the correlation, so rows 0..12 are 0.0
    // rather than NaN, and so is any later window min_periods or pandas'
    // constant-window guard rejected.
    t.put(codes::F_ACF_LAG1,
          fillNaZeroLocal(pdops::rollCorr(hist_return, pdops::shift(hist_return, 1),
                                          STATS_WINDOW - 1, STATS_WINDOW - 1)));

    // --- scale-free levels, features_scalefree.py:113-129 -------------------
    // Called from research.py:639-646 with the DEFAULT window (20) and
    // obv_is_cumulative=False. Emitted in the reference's dict order, which is
    // the order the concat sees (`for _, s in scale_free_levels(...).items()`).
    //
    // The seven sources are read back OUT OF THE TABLE rather than recomputed,
    // which is the reference's own rule ("DERIVED, NOT RECOMPUTED",
    // features_scalefree.py module docstring): agamotto's BBANDS is
    // timeperiod=20 while mjolnir's is the talib default 5, so a recomputation
    // here would silently impose one engine's parameters on the other.
    // `Table::get` throws on a missing name, so a TA-Lib block that stopped
    // emitting one of them fails loudly instead of dropping a column.
    {
        const std::vector<double>& close = bars.close;
        // COPIES, not references. `Table::put` push_backs onto `cols`, which
        // reallocates the outer vector and moves every inner vector out from
        // under any reference taken earlier — a use-after-free that would read
        // as a value mismatch on whichever column happened to land there. Only
        // sources held ACROSS a `put` need this; the single-use `t.get(...)`
        // calls below are consumed by `safeDiv` before their `put` runs.
        const std::vector<double> bb_upper = t.get(codes::F_BB_UPPER);
        const std::vector<double> bb_lower = t.get(codes::F_BB_LOWER);
        const std::vector<double> bb_span = subVec(bb_upper, bb_lower);

        // volume.rolling(20, min_periods=20).sum() on the RAW volume column —
        // the same unfilled volume the TA-Lib block gets, so a NaN volume bar
        // blanks the denominator for 20 rows on both sides.
        const std::vector<double> vol_sum =
            pdops::rollSum(bars.volume, SCALE_FREE_WINDOW, SCALE_FREE_WINDOW);

        t.put(codes::F_SAR_DIST, safeDiv(subVec(close, t.get(codes::F_SAR)), close));
        t.put(codes::F_BB_PCTB, safeDiv(subVec(close, bb_lower), bb_span));
        t.put(codes::F_BB_WIDTH, safeDiv(bb_span, close));
        t.put(codes::F_MACD_NORM, safeDiv(t.get(codes::F_MACD), close));
        t.put(codes::F_MACDHIST_NORM, safeDiv(t.get(codes::F_MACDHIST), close));

        // NO SECOND DIFFERENCE. `_flow(series, window, is_cumulative=False)`
        // returns the series UNCHANGED (features_scalefree.py:66-69); agamotto's
        // obv/ad are already `obv_raw.diff(14)` from research.py:538-539, so the
        // only thing left to do is normalise the UNITS. Differencing again is
        // silent — on a steady flow it returns ~0 every row and still looks like
        // a feature. Note the windows differ on purpose: the flow is a 14-bar
        // difference, the normaliser a 20-bar volume sum.
        t.put(codes::F_OBV_SLOPE, safeDiv(t.get(codes::F_OBV), vol_sum));
        t.put(codes::F_AD_SLOPE, safeDiv(t.get(codes::F_AD), vol_sum));
    }

    return t;
}

} // namespace agamotto
