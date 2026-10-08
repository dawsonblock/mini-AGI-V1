#pragma once

#include "qw3/device_backend.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace qw3 {

// Which projection a LoRA entry applies to. Output = the v1 LM-head
// target; Q/K/V/O are per-layer attention projections (v2 bundles).
enum class NativeLoraKind : uint8_t { Output = 0, Q, K, V, O };

inline const char *native_lora_kind_name(NativeLoraKind k) {
    switch (k) {
        case NativeLoraKind::Output: return "output.weight";
        case NativeLoraKind::Q: return "self_attn.q_proj";
        case NativeLoraKind::K: return "self_attn.k_proj";
        case NativeLoraKind::V: return "self_attn.v_proj";
        case NativeLoraKind::O: return "self_attn.o_proj";
    }
    return "unknown";
}

// Expected per-kind dimensions of the model the bundle serves, used to
// validate v2 attention entries at load time. q_rows/kv_rows are the
// projection output widths (n_heads*head_dim and n_kv_heads*head_dim).
struct NativeLoraAttentionDims {
    uint32_t n_layers = 0;
    uint32_t hidden = 0;   // n_embd — input width of q/k/v, output of o
    uint32_t q_rows = 0;   // output width of q_proj (includes gate width
                           // on gated-attention models)
    uint32_t kv_rows = 0;  // output width of k_proj and v_proj
    uint32_t o_in = 0;     // input width of o_proj — post-attention mid
                           // width (n_heads*head_dim); differs from
                           // q_rows on gated-attention models where the
                           // q weight fuses the output gate
};

struct NativeLoraHostEntry {
    std::string target;
    NativeLoraKind kind = NativeLoraKind::Output;
    // Attention entries only: which transformer layer. kNoLayer = output.
    static constexpr uint32_t kNoLayer = 0xffffffffu;
    uint32_t layer = kNoLayer;
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
// v1 bundles (NativeAdapter1) support only the LM-head target. v2 bundles
// (NativeAdapter2, schema qw3-native-lora-bundle-v2) may additionally carry
// per-layer self_attn.{q,k,v,o}_proj entries validated against the declared
// model dimensions. Unsupported targets are rejected at load rather than
// silently omitted.
class NativeLoraHostBundle {
public:
    // attn_dims: required when the bundle contains attention entries;
    // must be null for strict v1 loading (attention targets rejected).
    static NativeLoraHostBundle load(const std::string &directory,
                                     const std::string &expected_adapter_set_root,
                                     const std::string &expected_bundle_root,
                                     const std::string &expected_model_sha256,
                                     uint32_t expected_input_features,
                                     uint32_t expected_output_features,
                                     const NativeLoraAttentionDims *attn_dims = nullptr);

    const NativeLoraBundleInfo &info() const { return info_; }
    const std::vector<NativeLoraHostEntry> &entries() const { return entries_; }

    bool has_attention() const {
        for (const auto &e : entries_) {
            if (e.kind != NativeLoraKind::Output) return true;
        }
        return false;
    }

    // CPU oracle used by qualification tests. `out` is updated in place:
    // out += scale * B * (A * input) for every matching entry.
    void apply_output_cpu(const std::vector<float> &input,
                          std::vector<float> &out) const;
    void apply_projection_cpu(NativeLoraKind kind,
                              uint32_t layer,
                              const std::vector<float> &input,
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
    bool has_attention() const {
        for (const auto &e : entries_) {
            if (e.kind != NativeLoraKind::Output) return true;
        }
        return false;
    }
    // True when the bundle declares at least one attention entry on this
    // layer — used by the executor to fail closed on code paths that
    // cannot apply per-projection deltas (e.g. fused GDN layers).
    bool attention_covers(uint32_t layer) const {
        for (const auto &e : entries_) {
            if (e.kind != NativeLoraKind::Output && e.layer == layer)
                return true;
        }
        return false;
    }

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

    // v2: apply the delta for one attention projection (kind, layer) to an
    // already-computed projection output: proj_out += scale*B*(A*input).
    // No-op when no entry matches. `input` is the projection's own input
    // (post-norm hidden for q/k/v; post-attention mid for o).
    DeviceStatus apply_projection(DeviceBackend &backend,
                                  NativeLoraKind kind,
                                  uint32_t layer,
                                  DeviceTensor &proj_out,
                                  const DeviceTensor &input,
                                  DeviceTensor &rank_scratch,
                                  uint32_t batch,
                                  uint32_t input_stride,
                                  uint32_t output_stride) const;

private:
    struct DeviceEntry {
        std::string target;
        NativeLoraKind kind = NativeLoraKind::Output;
        uint32_t layer = NativeLoraHostEntry::kNoLayer;
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
