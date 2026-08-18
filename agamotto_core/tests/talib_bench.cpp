// Bounds the "kline built -> signal generated" span BEFORE the feature engine
// exists, by measuring its dominant cost: the 25 TA-Lib indicators agamotto
// computes plus the rolling std/skew/kurt/acf_lag1 block, over the same 700-bar
// window the reference uses (VOL_Q_WINDOW / load_data(limit=700)).
//
// This is a LOWER BOUND on Phase 2's per-bar cost, not a prediction of it: the
// ~60-column feature frame, the 33 regime predicates and the Ridge dot product
// are on top. It exists so the Phase 1 report can quote a measured number
// instead of an estimate.
//
// The indicator list and periods mirror the reference exactly.
#include <ta-lib/ta_libc.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

constexpr int N = 700;

double mean(const std::vector<double>& v)
{
	double s = 0.0;
	for (double x : v) s += x;
	return s / static_cast<double>(v.size());
}

} // namespace

int main(int argc, char** argv)
{
	const int iters = (argc > 1) ? std::atoi(argv[1]) : 2000;

	// Deterministic synthetic OHLCV — a random walk. Content is irrelevant to
	// timing; only the window length is.
	std::vector<double> c(N), h(N), l(N), o(N), v(N);
	double px = 64000.0;
	unsigned seed = 12345;
	for (int i = 0; i < N; ++i) {
		seed = seed * 1103515245u + 12345u;
		const double step = (static_cast<double>((seed >> 16) & 0x7fff) / 32767.0 - 0.5) * 20.0;
		o[i] = px;
		px += step;
		c[i] = px;
		h[i] = std::max(o[i], c[i]) + 2.0;
		l[i] = std::min(o[i], c[i]) - 2.0;
		v[i] = 10.0 + static_cast<double>((seed >> 8) & 0xff) / 10.0;
	}

	std::vector<double> out(N), out2(N), out3(N);
	int begin = 0, n = 0;

	std::vector<double> samples;
	samples.reserve(static_cast<size_t>(iters));

	for (int it = 0; it < iters; ++it) {
		const auto t0 = std::chrono::steady_clock::now();

		TA_RSI(0, N - 1, c.data(), 14, &begin, &n, out.data());
		TA_RSI(0, N - 1, c.data(), 7,  &begin, &n, out.data());
		TA_RSI(0, N - 1, c.data(), 28, &begin, &n, out.data());
		TA_MACD(0, N - 1, c.data(), 12, 26, 9, &begin, &n, out.data(), out2.data(), out3.data());
		TA_STOCH(0, N - 1, h.data(), l.data(), c.data(), 5, 3, TA_MAType_SMA, 3, TA_MAType_SMA,
		         &begin, &n, out.data(), out2.data());
		TA_CCI(0, N - 1, h.data(), l.data(), c.data(), 14, &begin, &n, out.data());
		TA_ADX(0, N - 1, h.data(), l.data(), c.data(), 14, &begin, &n, out.data());
		TA_DX(0, N - 1, h.data(), l.data(), c.data(), 14, &begin, &n, out.data());
		TA_PLUS_DI(0, N - 1, h.data(), l.data(), c.data(), 14, &begin, &n, out.data());
		TA_MINUS_DI(0, N - 1, h.data(), l.data(), c.data(), 14, &begin, &n, out.data());
		TA_MOM(0, N - 1, c.data(), 10, &begin, &n, out.data());
		TA_ROC(0, N - 1, c.data(), 10, &begin, &n, out.data());
		TA_WILLR(0, N - 1, h.data(), l.data(), c.data(), 14, &begin, &n, out.data());
		TA_CMO(0, N - 1, c.data(), 14, &begin, &n, out.data());
		TA_TRIX(0, N - 1, c.data(), 30, &begin, &n, out.data());
		TA_ULTOSC(0, N - 1, h.data(), l.data(), c.data(), 7, 14, 28, &begin, &n, out.data());
		TA_STOCHRSI(0, N - 1, c.data(), 14, 5, 3, TA_MAType_SMA, &begin, &n, out.data(), out2.data());
		TA_OBV(0, N - 1, c.data(), v.data(), &begin, &n, out.data());
		TA_AD(0, N - 1, h.data(), l.data(), c.data(), v.data(), &begin, &n, out.data());
		TA_MFI(0, N - 1, h.data(), l.data(), c.data(), v.data(), 14, &begin, &n, out.data());
		TA_BOP(0, N - 1, o.data(), h.data(), l.data(), c.data(), &begin, &n, out.data());
		TA_ATR(0, N - 1, h.data(), l.data(), c.data(), 14, &begin, &n, out.data());
		TA_NATR(0, N - 1, h.data(), l.data(), c.data(), 14, &begin, &n, out.data());
		TA_BBANDS(0, N - 1, c.data(), 20, 2.0, 2.0, TA_MAType_SMA, &begin, &n,
		          out.data(), out2.data(), out3.data());
		TA_SAR(0, N - 1, h.data(), l.data(), 0.02, 0.2, &begin, &n, out.data());

		// Rolling std / skew / kurt / acf_lag1 over STATS_WINDOW=14, and the
		// Parkinson vol, all computed the way the reference does.
		constexpr int W = 14;
		double acc = 0.0;
		for (int i = W; i < N; ++i) {
			double m = 0.0;
			for (int k = i - W; k < i; ++k) m += c[k];
			m /= W;
			double s2 = 0.0, s3 = 0.0, s4 = 0.0;
			for (int k = i - W; k < i; ++k) {
				const double d = c[k] - m;
				s2 += d * d; s3 += d * d * d; s4 += d * d * d * d;
			}
			const double sd = std::sqrt(s2 / (W - 1));
			acc += sd + s3 + s4;
			double num = 0.0, dx = 0.0, dy = 0.0;
			for (int k = i - W + 1; k < i; ++k) {
				num += (c[k] - m) * (c[k - 1] - m);
				dx += (c[k] - m) * (c[k] - m);
				dy += (c[k - 1] - m) * (c[k - 1] - m);
			}
			acc += (dx > 0 && dy > 0) ? num / std::sqrt(dx * dy) : 0.0;
			acc += std::sqrt(1.0 / (4.0 * std::log(2.0))
			                 * std::pow(std::log(h[i] / l[i]), 2.0));
		}

		const auto t1 = std::chrono::steady_clock::now();
		samples.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
		if (acc == 12345.6789) std::printf(" ");   // defeat dead-code elimination
	}

	std::sort(samples.begin(), samples.end());
	auto pct = [&](double q) {
		return samples[std::min(static_cast<size_t>(q * (samples.size() - 1) + 0.5),
		                        samples.size() - 1)];
	};
	std::printf("TA-Lib(25 indicators) + rolling stats over a %d-bar window, %d iterations\n",
	            N, iters);
	std::printf("  min %.1f us  p50 %.1f us  mean %.1f us  p99 %.1f us  max %.1f us\n",
	            samples.front(), pct(0.50), mean(samples), pct(0.99), samples.back());
	std::printf("  NOTE: lower bound on bar->signal. The ~60-column feature frame,\n"
	            "        33 regime predicates and the Ridge dot product are on top.\n");
	return 0;
}
