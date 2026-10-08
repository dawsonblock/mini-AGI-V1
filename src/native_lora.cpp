#include "qw3/native_lora.hpp"

#include "qw3/kvmem_archive.hpp"

#include "json.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>

namespace qw3 {
namespace {
using json = nlohmann::json;
namespace fs = std::filesystem;

bool is_hex64(const std::string &s) {
    if (s.size() != 64) return false;
    return std::all_of(s.begin(), s.end(), [](unsigned char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    });
}

bool is_governance_digest(const std::string &s) {
    return s.size() == 71 && s.rfind("sha256:", 0) == 0 &&
           is_hex64(s.substr(7));
}

fs::path safe_bundle_file(const fs::path &root, const std::string &rel) {
    fs::path p(rel);
    if (p.empty() || p.is_absolute()) {
        throw std::runtime_error("native LoRA payload path must be relative");
    }
    for (const auto &part : p) {
        if (part == "." || part == "..") {
            throw std::runtime_error("unsafe native LoRA payload path");
        }
    }
    const fs::path full = root / p;
    if (!fs::exists(full) || !fs::is_regular_file(full) || fs::is_symlink(full)) {
        throw std::runtime_error("native LoRA payload is missing or unsafe: " + rel);
    }
    const fs::path canonical_root = fs::weakly_canonical(root);
    const fs::path canonical_file = fs::weakly_canonical(full);
    auto r = canonical_root.begin();
    auto f = canonical_file.begin();
    for (; r != canonical_root.end(); ++r, ++f) {
        if (f == canonical_file.end() || *r != *f) {
            throw std::runtime_error("native LoRA payload escapes bundle root");
        }
    }
    return canonical_file;
}

std::vector<float> read_f32_file(const fs::path &path, uint64_t expected_bytes) {
    const uint64_t actual = static_cast<uint64_t>(fs::file_size(path));
    if (actual != expected_bytes || (actual % sizeof(float)) != 0) {
        throw std::runtime_error("native LoRA FP32 payload byte-size mismatch: " +
                                 path.string());
    }
    std::vector<float> out(static_cast<size_t>(actual / sizeof(float)));
    std::ifstream in(path, std::ios::binary);
    if (!in || (actual && !in.read(reinterpret_cast<char *>(out.data()),
                                   static_cast<std::streamsize>(actual)))) {
        throw std::runtime_error("failed to read native LoRA payload: " + path.string());
    }
    return out;
}

uint32_t require_u32(const json &j, const char *name) {
    const uint64_t value = j.at(name).get<uint64_t>();
    if (value == 0 || value > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error(std::string("invalid native LoRA ") + name);
    }
    return static_cast<uint32_t>(value);
}

} // namespace

NativeLoraHostBundle NativeLoraHostBundle::load(
        const std::string &directory,
        const std::string &expected_adapter_set_root,
        const std::string &expected_bundle_root,
        const std::string &expected_model_sha256,
        uint32_t expected_input_features,
        uint32_t expected_output_features,
        const NativeLoraAttentionDims *attn_dims) {
    if (!is_hex64(expected_adapter_set_root) ||
        !is_hex64(expected_bundle_root) ||
        !is_hex64(expected_model_sha256)) {
        throw std::runtime_error("native LoRA expected identities must be lowercase sha256 hex");
    }
    const fs::path root = fs::weakly_canonical(fs::path(directory));
    if (!fs::exists(root) || !fs::is_directory(root)) {
        throw std::runtime_error("native LoRA bundle directory does not exist: " + directory);
    }
    const fs::path manifest_path = safe_bundle_file(root, "manifest.json");
    const std::string bundle_root = kvmem_archive_model_sha256(manifest_path.string());
    if (bundle_root != expected_bundle_root) {
        throw std::runtime_error("native LoRA manifest digest does not match governed bundle root");
    }

    json manifest;
    {
        std::ifstream in(manifest_path);
        if (!in) throw std::runtime_error("cannot open native LoRA manifest");
        in >> manifest;
    }
    const std::string schema =
        manifest.value("schema", std::string());
    const bool is_v1 = schema == "qw3-native-lora-bundle-v1";
    const bool is_v2 = schema == "qw3-native-lora-bundle-v2";
    if (!is_v1 && !is_v2) {
        throw std::runtime_error("unsupported native LoRA bundle schema");
    }
    const std::string adapter_root = manifest.value("adapter_set_root", std::string());
    const std::string model_sha = manifest.value("model_sha256", std::string());
    const std::string foundation = manifest.value("foundation_model_digest", std::string());
    const std::string qualified = manifest.value("qualified_candidate_digest", std::string());
    if (adapter_root != expected_adapter_set_root) {
        throw std::runtime_error("native LoRA adapter-set root mismatch");
    }
    if (model_sha != expected_model_sha256) {
        throw std::runtime_error("native LoRA model SHA-256 mismatch");
    }
    if (!is_governance_digest(foundation) || !is_governance_digest(qualified)) {
        throw std::runtime_error("native LoRA governance provenance is malformed");
    }

    const auto &tensors = manifest.at("tensors");
    if (!tensors.is_array() || tensors.empty()) {
        throw std::runtime_error("native LoRA bundle has no tensors");
    }

    NativeLoraHostBundle out;
    out.info_.schema = schema;
    out.info_.adapter_set_root = adapter_root;
    out.info_.bundle_root = bundle_root;
    out.info_.model_sha256 = model_sha;
    out.info_.foundation_model_digest = foundation;
    out.info_.qualified_candidate_digest = qualified;

    std::set<std::pair<NativeLoraKind, uint32_t>> seen_targets;
    for (const auto &item : tensors) {
        NativeLoraHostEntry entry;
        entry.target = item.at("target").get<std::string>();
        entry.rank = require_u32(item, "rank");
        entry.in_features = require_u32(item, "in_features");
        entry.out_features = require_u32(item, "out_features");
        entry.scale = item.at("scale").get<float>();
        if (!std::isfinite(entry.scale) || entry.scale == 0.0f) {
            throw std::runtime_error("native LoRA scale must be finite and non-zero");
        }

        if (entry.target == "output.weight") {
            entry.kind = NativeLoraKind::Output;
            entry.layer = NativeLoraHostEntry::kNoLayer;
            if (entry.in_features != expected_input_features ||
                entry.out_features != expected_output_features) {
                throw std::runtime_error(
                    "native LoRA output.weight dimensions do not match model");
            }
        } else if (is_v2 &&
                   (entry.target == "self_attn.q_proj" ||
                    entry.target == "self_attn.k_proj" ||
                    entry.target == "self_attn.v_proj" ||
                    entry.target == "self_attn.o_proj")) {
            // v2 attention entries: require declared model geometry and
            // an explicit layer index within range.
            if (attn_dims == nullptr) {
                throw std::runtime_error(
                    "native LoRA attention entries require declared model dims");
            }
            if (entry.target == "self_attn.q_proj")
                entry.kind = NativeLoraKind::Q;
            else if (entry.target == "self_attn.k_proj")
                entry.kind = NativeLoraKind::K;
            else if (entry.target == "self_attn.v_proj")
                entry.kind = NativeLoraKind::V;
            else
                entry.kind = NativeLoraKind::O;
            const uint64_t layer_v = item.at("layer").get<uint64_t>();
            if (layer_v > std::numeric_limits<uint32_t>::max()) {
                throw std::runtime_error("invalid native LoRA layer");
            }
            entry.layer = static_cast<uint32_t>(layer_v);
            if (entry.layer >= attn_dims->n_layers) {
                throw std::runtime_error(
                    "native LoRA layer index exceeds model layers: " +
                    entry.target);
            }
            const uint32_t want_in = (entry.kind == NativeLoraKind::O)
                ? attn_dims->o_in : attn_dims->hidden;
            const uint32_t want_out = (entry.kind == NativeLoraKind::O)
                ? attn_dims->hidden
                : (entry.kind == NativeLoraKind::Q ? attn_dims->q_rows
                                                 : attn_dims->kv_rows);
            if (entry.in_features != want_in || entry.out_features != want_out) {
                throw std::runtime_error(
                    "native LoRA attention dimensions do not match model: " +
                    entry.target);
            }
        } else {
            throw std::runtime_error(
                "native LoRA unsupported target: " + entry.target);
        }
        if (!seen_targets.insert({entry.kind, entry.layer}).second) {
            throw std::runtime_error(
                "native LoRA duplicate target entry: " + entry.target +
                " layer " + std::to_string(entry.layer));
        }

        const auto &aj = item.at("a");
        const auto &bj = item.at("b");
        const std::string afile = aj.at("file").get<std::string>();
        const std::string bfile = bj.at("file").get<std::string>();
        const std::string asha = aj.at("sha256").get<std::string>();
        const std::string bsha = bj.at("sha256").get<std::string>();
        const uint64_t abytes = aj.at("bytes").get<uint64_t>();
        const uint64_t bbytes = bj.at("bytes").get<uint64_t>();
        const uint64_t want_a = static_cast<uint64_t>(entry.rank) * entry.in_features * sizeof(float);
        const uint64_t want_b = static_cast<uint64_t>(entry.out_features) * entry.rank * sizeof(float);
        if (!is_hex64(asha) || !is_hex64(bsha) || abytes != want_a || bbytes != want_b) {
            throw std::runtime_error("native LoRA tensor metadata is inconsistent with shape");
        }
        const fs::path apath = safe_bundle_file(root, afile);
        const fs::path bpath = safe_bundle_file(root, bfile);
        if (kvmem_archive_model_sha256(apath.string()) != asha ||
            kvmem_archive_model_sha256(bpath.string()) != bsha) {
            throw std::runtime_error("native LoRA tensor digest mismatch");
        }
        entry.a = read_f32_file(apath, abytes);
        entry.b = read_f32_file(bpath, bbytes);
        out.info_.max_rank = std::max(out.info_.max_rank, entry.rank);
        out.entries_.push_back(std::move(entry));
    }
    out.info_.tensor_count = static_cast<uint32_t>(out.entries_.size());
    return out;
}

namespace {
void lora_accumulate_cpu(const NativeLoraHostEntry &entry,
                         const std::vector<float> &input,
                         std::vector<float> &out) {
    if (input.size() != entry.in_features || out.size() != entry.out_features) {
        throw std::invalid_argument("native LoRA CPU oracle shape mismatch");
    }
    std::vector<float> rank(entry.rank, 0.0f);
    for (uint32_t r = 0; r < entry.rank; ++r) {
        double sum = 0.0;
        for (uint32_t c = 0; c < entry.in_features; ++c) {
            sum += static_cast<double>(entry.a[static_cast<size_t>(r) * entry.in_features + c]) *
                   static_cast<double>(input[c]);
        }
        rank[r] = static_cast<float>(sum);
    }
    for (uint32_t row = 0; row < entry.out_features; ++row) {
        double sum = 0.0;
        for (uint32_t r = 0; r < entry.rank; ++r) {
            sum += static_cast<double>(entry.b[static_cast<size_t>(row) * entry.rank + r]) *
                   static_cast<double>(rank[r]);
        }
        out[row] += entry.scale * static_cast<float>(sum);
    }
}
} // namespace

void NativeLoraHostBundle::apply_output_cpu(
        const std::vector<float> &input,
        std::vector<float> &out) const {
    for (const auto &entry : entries_) {
        if (entry.kind != NativeLoraKind::Output) continue;
        lora_accumulate_cpu(entry, input, out);
    }
}

void NativeLoraHostBundle::apply_projection_cpu(
        NativeLoraKind kind,
        uint32_t layer,
        const std::vector<float> &input,
        std::vector<float> &out) const {
    for (const auto &entry : entries_) {
        if (entry.kind != kind || entry.layer != layer) continue;
        lora_accumulate_cpu(entry, input, out);
    }
}

std::shared_ptr<NativeLoraSet> NativeLoraSet::upload(
        const NativeLoraHostBundle &host,
        DeviceBackend &backend) {
    auto out = std::make_shared<NativeLoraSet>();
    out->info_ = host.info();
    out->entries_.reserve(host.entries().size());
    for (const auto &src : host.entries()) {
        DeviceEntry dst;
        dst.target = src.target;
        dst.kind = src.kind;
        dst.layer = src.layer;
        dst.rank = src.rank;
        dst.in_features = src.in_features;
        dst.out_features = src.out_features;
        dst.scale = src.scale;
        dst.a = backend.weight_f32(src.a.data(), src.a.size(), "native_lora.A");
        dst.b = backend.weight_f32(src.b.data(), src.b.size(), "native_lora.B");
        if (!dst.a || !dst.b) throw std::runtime_error("failed to upload native LoRA weights");
        dst.a->rows = src.rank;
        dst.a->cols = src.in_features;
        dst.b->rows = src.out_features;
        dst.b->cols = src.rank;
        out->entries_.push_back(std::move(dst));
    }
    return out;
}

DeviceStatus NativeLoraSet::apply_output(
        DeviceBackend &backend,
        DeviceTensor &logits,
        const DeviceTensor &normalized_hidden,
        DeviceTensor &rank_scratch,
        uint32_t batch,
        uint32_t input_stride,
        uint32_t output_stride) const {
    if (entries_.empty() || batch == 0) return {};
    if (rank_scratch.count < static_cast<uint64_t>(batch) * info_.max_rank) {
        return {false, "native LoRA rank scratch is too small"};
    }
    for (const auto &entry : entries_) {
        if (entry.kind != NativeLoraKind::Output) continue;
        if (input_stride < entry.in_features || output_stride < entry.out_features) {
            return {false, "native LoRA runtime stride mismatch"};
        }
        if (auto st = backend.dense_f32_matmul(
                rank_scratch, *entry.a, normalized_hidden,
                batch, input_stride, info_.max_rank,
                1.0f, false); !st.ok) {
            return st;
        }
        if (auto st = backend.dense_f32_matmul(
                logits, *entry.b, rank_scratch,
                batch, info_.max_rank, output_stride,
                entry.scale, true); !st.ok) {
            return st;
        }
    }
    return {};
}

DeviceStatus NativeLoraSet::apply_projection(
        DeviceBackend &backend,
        NativeLoraKind kind,
        uint32_t layer,
        DeviceTensor &proj_out,
        const DeviceTensor &input,
        DeviceTensor &rank_scratch,
        uint32_t batch,
        uint32_t input_stride,
        uint32_t output_stride) const {
    if (entries_.empty() || batch == 0) return {};
    if (rank_scratch.count < static_cast<uint64_t>(batch) * info_.max_rank) {
        return {false, "native LoRA rank scratch is too small"};
    }
    for (const auto &entry : entries_) {
        if (entry.kind != kind || entry.layer != layer) continue;
        if (input_stride < entry.in_features || output_stride < entry.out_features) {
            return {false, "native LoRA runtime stride mismatch"};
        }
        if (auto st = backend.dense_f32_matmul(
                rank_scratch, *entry.a, input,
                batch, input_stride, info_.max_rank,
                1.0f, false); !st.ok) {
            return st;
        }
        if (auto st = backend.dense_f32_matmul(
                proj_out, *entry.b, rank_scratch,
                batch, info_.max_rank, output_stride,
                entry.scale, true); !st.ok) {
            return st;
        }
    }
    return {};
}

} // namespace qw3
