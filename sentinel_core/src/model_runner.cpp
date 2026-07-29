#include "model_runner.hpp"
#include "regime_gate.hpp"          // FeaturePanel
#include "feature_map_generated.hpp"

#include <LightGBM/c_api.h>

#include <cmath>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace mjolnir {

std::string encodeFeatureName(const std::string& real)
{
    const auto& M = featureNameToCode();
    // Exact hit first.
    auto it = M.find(real);
    if (it != M.end()) return it->second;

    // Structure-preserving: "<base>_roll<w>_<stat>" -> "<code>_roll<w>_<stat>".
    // The codec keeps this suffix as structure rather than mapping the whole
    // composed name, so the base must be encoded and the suffix reattached.
    static const std::regex re(R"(^(.*)_roll(\d+)_(mean|std)$)");
    std::smatch m;
    if (std::regex_match(real, m, re)) {
        auto bit = M.find(m[1].str());
        if (bit != M.end()) return bit->second + "_roll" + m[2].str() + "_" + m[3].str();
    }
    return real;   // unmapped (passthrough column) — caller decides if that is OK
}

ModelRunner::~ModelRunner()
{
    if (mBooster) LGBM_BoosterFree(mBooster);
}

void ModelRunner::load(const std::string& regime_dir)
{
    const std::string model_p  = regime_dir + "/model.txt";
    const std::string scaler_p = regime_dir + "/scaler.txt";
    const std::string feats_p  = regime_dir + "/features.txt";

    {
        std::ifstream fh(feats_p);
        if (!fh) throw std::runtime_error("cannot open " + feats_p);
        std::string line;
        while (std::getline(fh, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (!line.empty()) mFeatures.push_back(line);
        }
    }
    if (mFeatures.empty()) throw std::runtime_error("no features in " + feats_p);

    {
        std::ifstream fh(scaler_p);
        if (!fh) throw std::runtime_error("cannot open " + scaler_p);
        size_t n = 0;
        fh >> n;
        if (n != mFeatures.size())
            throw std::runtime_error("scaler length != feature count in " + scaler_p);
        mCenter.resize(n);
        mScale.resize(n);
        for (size_t i = 0; i < n; ++i) {
            if (!(fh >> mCenter[i] >> mScale[i]))
                throw std::runtime_error("truncated scaler at row " + std::to_string(i));
            // A zero scale would divide by zero and yield inf for every row.
            if (mScale[i] == 0.0) mScale[i] = 1.0;
        }
    }

    int out_iter = 0;
    if (LGBM_BoosterCreateFromModelfile(model_p.c_str(), &out_iter, &mBooster) != 0)
        throw std::runtime_error("LGBM_BoosterCreateFromModelfile failed: " + model_p);
    mNumIteration = out_iter;

    int n_model_feat = 0;
    if (LGBM_BoosterGetNumFeature(mBooster, &n_model_feat) != 0)
        throw std::runtime_error("LGBM_BoosterGetNumFeature failed");
    if (static_cast<size_t>(n_model_feat) != mFeatures.size())
        throw std::runtime_error(
            "model expects " + std::to_string(n_model_feat) + " features but features.txt has "
            + std::to_string(mFeatures.size()));
}

double ModelRunner::predictRow(const FeaturePanel& panel, size_t row) const
{
    if (!mBooster) throw std::runtime_error("ModelRunner: not loaded");

    // Build the row in MODEL feature order, mapping the panel's real names to
    // the coded names the model was trained on. Build the lookup once per call;
    // the incremental design will hoist this to load() time.
    std::vector<double> x(mFeatures.size());
    for (size_t j = 0; j < mFeatures.size(); ++j) {
        const std::string& want = mFeatures[j];
        double v = 0.0;
        bool found = false;
        // Fast path: the panel may already carry the coded name.
        if (panel.has(want)) {
            v = panel.get(want)[row];
            found = true;
        } else {
            // Otherwise scan for the real name that encodes to it.
            for (const auto& real : panel.names()) {
                if (encodeFeatureName(real) == want) {
                    v = panel.get(real)[row];
                    found = true;
                    break;
                }
            }
        }
        if (!found)
            throw std::runtime_error(
                "model feature '" + want + "' not present in the panel. Substituting a "
                "default would silently change the prediction.");
        // The reference replaces inf with NaN then NaN with 0.0 before scaling.
        if (std::isinf(v) || std::isnan(v)) v = 0.0;
        x[j] = (v - mCenter[j]) / mScale[j];
    }

    double out = 0.0;
    int64_t out_len = 0;
    if (LGBM_BoosterPredictForMat(
            mBooster, x.data(), C_API_DTYPE_FLOAT64, 1, static_cast<int32_t>(x.size()),
            /*is_row_major=*/1, C_API_PREDICT_NORMAL, 0, mNumIteration, "", &out_len, &out) != 0)
        throw std::runtime_error("LGBM_BoosterPredictForMat failed");
    return out;
}

} // namespace mjolnir
