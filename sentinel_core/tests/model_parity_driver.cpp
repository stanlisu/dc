// Model-runner parity: load the EXPORTED weight dir, predict the same raw
// feature rows the reference scored, and compare.
//
// This is stronger than matching a LightGBM version string: it checks the
// property that actually matters — that the C++ path reproduces the deployed
// sklearn+pickle path numerically.
//
// Usage: model_parity_driver <regime_dir> <ref_preds.txt>
//   ref file: "<n_feats>", then N rows of raw features, then N reference preds.
#include "../src/model_runner.hpp"
#include "../src/regime_gate.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace mjolnir;

int main(int argc, char** argv)
{
    if (argc < 3) { std::fprintf(stderr, "usage: %s <regime_dir> <ref.txt>\n", argv[0]); return 2; }
    const std::string dir = argv[1];

    ModelRunner mr;
    mr.load(dir);

    std::ifstream fh(argv[2]);
    if (!fh) { std::fprintf(stderr, "cannot open %s\n", argv[2]); return 2; }
    size_t nf = 0;
    fh >> nf;
    if (nf != mr.featureCount()) {
        std::fprintf(stderr, "feature count mismatch: ref=%zu model=%zu\n", nf, mr.featureCount());
        return 1;
    }

    // Read all remaining numbers: N rows x nf, then N preds. N is inferred.
    std::vector<double> nums;
    double v;
    while (fh >> v) nums.push_back(v);
    const size_t n_rows = nums.size() / (nf + 1);
    if (n_rows == 0 || nums.size() != n_rows * (nf + 1)) {
        std::fprintf(stderr, "malformed ref file\n");
        return 1;
    }

    // The runner selects features from a FeaturePanel by coded name; build a
    // one-row panel whose column names ARE the model's coded feature names, so
    // this isolates the scaler+booster path from feature-name mapping (which
    // the feature-parity harness covers separately).
    const std::vector<std::string>& names = mr.features();

    double max_abs = 0.0, max_rel = 0.0;
    for (size_t r = 0; r < n_rows; ++r) {
        std::vector<std::vector<double>> cols(nf, std::vector<double>(1, 0.0));
        for (size_t j = 0; j < nf; ++j) cols[j][0] = nums[r * nf + j];
        FeaturePanel panel(names, cols);
        const double got = mr.predictRow(panel, 0);
        const double ref = nums[n_rows * nf + r];
        const double a = std::fabs(got - ref);
        const double rel = a / std::max(1e-12, std::fabs(ref));
        if (a > max_abs) max_abs = a;
        if (rel > max_rel) max_rel = rel;
    }

    std::printf("[model_parity] rows=%zu feats=%zu max_abs_diff=%.6e max_rel_diff=%.6e\n",
                n_rows, nf, max_abs, max_rel);
    if (max_abs > 1e-9) {
        std::printf("=== FAIL: C++ predictions differ from the deployed model ===\n");
        return 1;
    }
    std::printf("=== PASS: predictions identical to the deployed model ===\n");
    return 0;
}
