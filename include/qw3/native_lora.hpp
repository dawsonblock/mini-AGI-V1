#pragma once

#include "qw3/device_backend.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace qw3 {

struct NativeLoraHostEntry {
    std::string target;
    uint32_t rank = 0;
    uint32_t in_features = 0;
    uint32_t out_features = 0;
    float scale = 1.0f;
    std::vector<float> a; // [rank, in_features], row-major
    std::vector<float> b; // [out_features, rank], row-major
};

struct NativeLoraBundleInfo {
    std::string schema;
    std::string adapter_set_root;
    std::string bundle_root;
    std::string model_sha256;
    std::string foundation_model_digest;
    std::string qualified_candidate_digest;
    uint32_t tensor_count = 0;
    uint32_t max_rank = 0;
};

// Portable, byte-verified host representation of a QW3 native LoRA bundle.
// NativeAdapter1 deliberately supports only the LM-head target. Unsupported
// targets are rejected at load rather than silently omitted.
class NativeLoraHostBundle {
public:
    static NativeLoraHostBundle load(const std::string &directory,
                                     const std::string &expected_adapter_set_root,
                                     const std::string &expected_bundle_root,
                                     const std::string &expected_model_sha256,
                                     uint32_t expected_input_features,
                                     uint32_t expected_output_features);

    const NativeLoraBundleInfo &info() const { return info_; }
    const std::vector<NativeLoraHostEntry> &entries() const { return entries_; }

    // CPU oracle used by qualification tests. `out` is updated in place:
    // out += scale * B * (A * input) for every entry in the bundle.
    void apply_output_cpu(const std::vector<float> &input,
                          std::vector<float> &out) const;

private:
    NativeLoraBundleInfo info_;
    std::vector<NativeLoraHostEntry> entries_;
};

class NativeLoraSet {
public:
    static std::shared_ptr<NativeLoraSet> upload(
        const NativeLoraHostBundle &host,
        DeviceBackend &backend);

    const NativeLoraBundleInfo &info() const { return info_; }
    uint32_t max_rank() const { return info_.max_rank; }
    bool empty() const { return entries_.empty(); }

    // Apply the qualified LM-head low-rank delta to an already-computed logits
    // tensor. Input must be FP32 normalized hidden state. `rank_scratch` must
    // have at least batch*max_rank elements.
    DeviceStatus apply_output(DeviceBackend &backend,
                              DeviceTensor &logits,
                              const DeviceTensor &normalized_hidden,
                              DeviceTensor &rank_scratch,
                              uint32_t batch,
                              uint32_t input_stride,
                              uint32_t output_stride) const;

private:
    struct DeviceEntry {
        std::string target;
        uint32_t rank = 0;
        uint32_t in_features = 0;
        uint32_t out_features = 0;
        float scale = 1.0f;
        std::unique_ptr<DeviceWeight> a;
        std::unique_ptr<DeviceWeight> b;
    };

    NativeLoraBundleInfo info_;
    std::vector<DeviceEntry> entries_;
};

} // namespace qw3
