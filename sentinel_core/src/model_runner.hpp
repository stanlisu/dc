#pragma once
// Model runner. PRIVATE.
//
// Loads a converted weight directory (model.txt / scaler.txt / features.txt —
// produced by tools/export_weights.py) and predicts through the LightGBM C API,
// i.e. the SAME inference kernel the Python wrapper calls. Predictions are
// therefore identical by construction rather than reimplemented.
//
// Two things here are easy to get silently wrong:
//   * the deployed scaler is a RobustScaler: (x - center) / scale. It is NOT
//     StandardScaler, and it has no mean_ attribute.
//   * the model's feature list is CODED (fNNN, with the _roll{w}_{stat} suffix
//     preserved), while the feature engine emits REAL names. Selection encodes
//     real -> code; a mismatch must raise, never silently pass a zero column.
#include <string>
#include <vector>

namespace mjolnir {

class FeaturePanel;

// real feature name -> coded name, preserving the _roll{w}_{mean,std} suffix.
std::string encodeFeatureName(const std::string& real);

class ModelRunner {
  public:
    ModelRunner() = default;
    ~ModelRunner();
    ModelRunner(const ModelRunner&) = delete;
    ModelRunner& operator=(const ModelRunner&) = delete;

    // Throws on any missing/!malformed artifact — a half-loaded model that
    // predicts from defaults is worse than a crash.
    void load(const std::string& regime_dir);

    // Predict for ONE row of the panel (the reference scores row iloc[-2]).
    // Selects this model's coded features from the panel, applies the scaler,
    // and runs the booster.
    double predictRow(const FeaturePanel& panel, size_t row) const;

    size_t featureCount() const { return mFeatures.size(); }
    const std::vector<std::string>& features() const { return mFeatures; }

  private:
    void* mBooster{nullptr};       // BoosterHandle
    int mNumIteration{0};
    std::vector<std::string> mFeatures;   // coded, model input order
    std::vector<double> mCenter;
    std::vector<double> mScale;
};

} // namespace mjolnir
