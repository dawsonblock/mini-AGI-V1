#include "qw3/gdn_runtime_bundle.hpp"
#include "qw3/gdn_summary_runtime.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <type_traits>

namespace qw3 {
namespace {

constexpr char kMagic[8] = {'Q','G','D','N','B','N','D','1'};
constexpr uint32_t kMaxStrings = 1u << 20;
constexpr uint32_t kMaxSummaries = 1u << 20;
constexpr uint64_t kMaxElements = uint64_t{1} << 31;
constexpr uint64_t kMaxPayload = uint64_t{8} << 30;

uint64_t fnv1a64(const uint8_t* data, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) {
        h ^= data[i];
        h *= 1099511628211ull;
    }
    return h;
}

class Writer {
public:
    template <class T> void pod(T v) {
        static_assert(std::is_trivially_copyable_v<T>);
        const auto* p = reinterpret_cast<const uint8_t*>(&v);
        b.insert(b.end(), p, p + sizeof(T));
    }
    void str(const std::string& s) {
        if (s.size() > kMaxStrings) throw std::invalid_argument("GDN bundle string too large");
        pod<uint32_t>(static_cast<uint32_t>(s.size()));
        b.insert(b.end(), s.begin(), s.end());
    }
    void f32(const std::vector<float>& v) {
        if (v.size() > kMaxElements) throw std::invalid_argument("GDN bundle vector too large");
        pod<uint64_t>(static_cast<uint64_t>(v.size()));
        const auto* p = reinterpret_cast<const uint8_t*>(v.data());
        b.insert(b.end(), p, p + v.size() * sizeof(float));
    }
    std::vector<uint8_t> b;
};

class Reader {
public:
    explicit Reader(std::vector<uint8_t> bytes) : b(std::move(bytes)) {}
    template <class T> T pod() {
        static_assert(std::is_trivially_copyable_v<T>);
        need(sizeof(T));
        T v{};
        std::memcpy(&v, b.data() + off, sizeof(T));
        off += sizeof(T);
        return v;
    }
    std::string str() {
        const uint32_t n = pod<uint32_t>();
        if (n > kMaxStrings) throw std::runtime_error("GDN bundle string length rejected");
        need(n);
        std::string out(reinterpret_cast<const char*>(b.data() + off), n);
        off += n;
        return out;
    }
    std::vector<float> f32() {
        const uint64_t n = pod<uint64_t>();
        if (n > kMaxElements || n > std::numeric_limits<size_t>::max() / sizeof(float)) {
            throw std::runtime_error("GDN bundle vector length rejected");
        }
        const size_t count = static_cast<size_t>(n);
        need(count * sizeof(float));
        std::vector<float> out(count);
        if (count) std::memcpy(out.data(), b.data() + off, count * sizeof(float));
        off += count * sizeof(float);
        return out;
    }
    bool done() const { return off == b.size(); }
private:
    void need(size_t n) const {
        if (n > b.size() - off) throw std::runtime_error("truncated GDN bundle payload");
    }
    std::vector<uint8_t> b;
    size_t off = 0;
};

void require_identity(const KvMemCacheIdentity& id) {
    if (id.base_model_digest.empty() || id.adapter_set_digest.empty() ||
        id.tokenizer_digest.empty() || id.layer_layout_digest.empty() ||
        id.position_scheme.empty() || id.recurrence_impl.empty()) {
        throw std::invalid_argument("GDN runtime bundle requires a complete cache identity");
    }
}

using BlockGeom = std::pair<uint64_t, uint32_t>;

std::map<uint32_t, std::vector<const GdnBlockAffineSummary*>> group_layers(
    const GdnRuntimeBundle& bundle) {
    std::map<uint32_t, std::vector<const GdnBlockAffineSummary*>> layers;
    for (const auto& s : bundle.summaries) layers[s.layer_index].push_back(&s);
    for (auto& [layer, vec] : layers) {
        (void)layer;
        std::sort(vec.begin(), vec.end(), [](auto* a, auto* b) {
            if (a->position_start != b->position_start) return a->position_start < b->position_start;
            return a->token_count < b->token_count;
        });
    }
    return layers;
}

std::vector<uint8_t> read_payload(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open GDN runtime bundle: " + path);
    char magic[8]{};
    uint64_t n = 0, expected = 0;
    in.read(magic, 8);
    in.read(reinterpret_cast<char*>(&n), sizeof(n));
    in.read(reinterpret_cast<char*>(&expected), sizeof(expected));
    if (!in || std::memcmp(magic, kMagic, 8) != 0) {
        throw std::runtime_error("invalid GDN runtime bundle header");
    }
    if (n > kMaxPayload || n > std::numeric_limits<size_t>::max()) {
        throw std::runtime_error("GDN runtime bundle payload length rejected");
    }
    std::vector<uint8_t> payload(static_cast<size_t>(n));
    if (n) in.read(reinterpret_cast<char*>(payload.data()), static_cast<std::streamsize>(n));
    if (!in) throw std::runtime_error("truncated GDN runtime bundle payload");
    char trailing = 0;
    if (in.read(&trailing, 1)) throw std::runtime_error("GDN runtime bundle has trailing bytes");
    if (fnv1a64(payload.data(), payload.size()) != expected) {
        throw std::runtime_error("GDN runtime bundle checksum mismatch");
    }
    return payload;
}

void write_payload(const std::string& path, const std::vector<uint8_t>& payload) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot open GDN runtime bundle output: " + path);
    const uint64_t n = payload.size();
    const uint64_t hash = fnv1a64(payload.data(), payload.size());
    out.write(kMagic, 8);
    out.write(reinterpret_cast<const char*>(&n), sizeof(n));
    out.write(reinterpret_cast<const char*>(&hash), sizeof(hash));
    if (n) out.write(reinterpret_cast<const char*>(payload.data()), static_cast<std::streamsize>(n));
    if (!out) throw std::runtime_error("failed writing GDN runtime bundle: " + path);
}

} // namespace

void GdnRuntimeBundle::validate() const {
    if (version != 1 && version != 2) throw std::invalid_argument("unsupported GDN runtime bundle version");
    if (model_id.empty()) throw std::invalid_argument("GDN runtime bundle model_id is empty");
    require_identity(identity);
    if (summaries.empty()) throw std::invalid_argument("GDN runtime bundle is empty");
    if (summaries.size() > kMaxSummaries) throw std::invalid_argument("too many GDN runtime summaries");

    const auto layers = group_layers(*this);
    if (layers.empty()) throw std::invalid_argument("GDN runtime bundle has no recurrent layers");

    std::vector<BlockGeom> canonical;
    bool first_layer = true;
    for (const auto& [layer, vec] : layers) {
        (void)layer;
        if (vec.empty()) throw std::invalid_argument("GDN runtime layer is empty");
        std::vector<BlockGeom> geometry;
        geometry.reserve(vec.size());
        uint64_t previous_end = 0;
        bool first = true;
        for (const auto* s : vec) {
            s->validate();
            const uint32_t required_summary_version = version >= 2 ? 3u : 2u;
            if (s->version != required_summary_version || s->model_id != model_id) {
                throw std::invalid_argument("GDN runtime bundle summary model/version mismatch");
            }
            if (!first && s->position_start < previous_end) {
                throw std::invalid_argument("GDN runtime bundle has overlapping/reordered blocks");
            }
            if (s->position_start > std::numeric_limits<uint64_t>::max() - s->token_count) {
                throw std::invalid_argument("GDN runtime block position overflow");
            }
            previous_end = s->position_start + s->token_count;
            first = false;
            geometry.emplace_back(s->position_start, s->token_count);
        }
        if (first_layer) {
            canonical = std::move(geometry);
            first_layer = false;
        } else if (geometry != canonical) {
            throw std::invalid_argument("GDN runtime recurrent layers do not share exact block geometry");
        }
    }
}

std::vector<uint32_t> GdnRuntimeBundle::layer_indices() const {
    validate();
    std::vector<uint32_t> out;
    uint32_t last = 0;
    bool first = true;
    for (const auto& s : summaries) {
        if (first || s.layer_index != last) {
            out.push_back(s.layer_index);
            last = s.layer_index;
            first = false;
        }
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

uint32_t GdnRuntimeBundle::block_count() const {
    validate();
    const auto layers = group_layers(*this);
    return static_cast<uint32_t>(layers.begin()->second.size());
}

GdnRuntimeBundle build_gdn_runtime_bundle(
    const KvMemCacheIdentity& identity,
    const std::vector<GdnBlockAffineSummary>& summaries) {
    if (summaries.empty()) throw std::invalid_argument("cannot build empty GDN runtime bundle");
    GdnRuntimeBundle out;
    const bool all_v3 = std::all_of(summaries.begin(), summaries.end(), [](const auto& s) { return s.version == 3; });
    const bool all_v2 = std::all_of(summaries.begin(), summaries.end(), [](const auto& s) { return s.version == 2; });
    if (!all_v3 && !all_v2) {
        throw std::invalid_argument("GDN runtime bundle cannot mix summary generations");
    }
    out.version = all_v3 ? 2u : 1u;
    out.identity = identity;
    out.summaries = summaries;
    std::sort(out.summaries.begin(), out.summaries.end(), [](const auto& a, const auto& b) {
        if (a.layer_index != b.layer_index) return a.layer_index < b.layer_index;
        if (a.position_start != b.position_start) return a.position_start < b.position_start;
        return a.token_count < b.token_count;
    });
    out.model_id = out.summaries.front().model_id;
    out.validate();
    return out;
}

void write_gdn_runtime_bundle(const std::string& path,
                              const GdnRuntimeBundle& bundle) {
    bundle.validate();
    Writer w;
    w.pod<uint32_t>(bundle.version);
    w.str(bundle.model_id);
    w.str(bundle.identity.base_model_digest);
    w.str(bundle.identity.adapter_set_digest);
    w.str(bundle.identity.tokenizer_digest);
    w.str(bundle.identity.layer_layout_digest);
    w.str(bundle.identity.position_scheme);
    w.str(bundle.identity.recurrence_impl);
    w.pod<uint32_t>(static_cast<uint32_t>(bundle.summaries.size()));
    for (const auto& s : bundle.summaries) {
        w.pod<uint32_t>(s.version);
        w.pod<uint32_t>(s.layer_index);
        w.pod<uint64_t>(s.position_start);
        w.pod<uint32_t>(s.token_count);
        w.pod<uint32_t>(s.num_v_heads);
        w.pod<uint32_t>(s.head_dim);
        w.str(s.model_id);
        w.str(s.producer);
        w.f32(s.T);
        w.f32(s.Z);
        if (bundle.version >= 2) {
            if (s.version < 3) throw std::invalid_argument("GDN bundle v2 requires v3 summaries");
            w.pod<uint32_t>(s.conv_kernel_size);
            w.pod<uint32_t>(s.conv_dim);
            w.pod<uint32_t>(s.conv_tail_rows);
            w.f32(s.conv_tail);
        }
    }
    write_payload(path, w.b);
}

GdnRuntimeBundle read_gdn_runtime_bundle(const std::string& path) {
    Reader r(read_payload(path));
    GdnRuntimeBundle out;
    out.version = r.pod<uint32_t>();
    out.model_id = r.str();
    out.identity.base_model_digest = r.str();
    out.identity.adapter_set_digest = r.str();
    out.identity.tokenizer_digest = r.str();
    out.identity.layer_layout_digest = r.str();
    out.identity.position_scheme = r.str();
    out.identity.recurrence_impl = r.str();
    const uint32_t n = r.pod<uint32_t>();
    if (n == 0 || n > kMaxSummaries) throw std::runtime_error("GDN runtime summary count rejected");
    out.summaries.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
        GdnBlockAffineSummary s;
        s.version = r.pod<uint32_t>();
        s.layer_index = r.pod<uint32_t>();
        s.position_start = r.pod<uint64_t>();
        s.token_count = r.pod<uint32_t>();
        s.num_v_heads = r.pod<uint32_t>();
        s.head_dim = r.pod<uint32_t>();
        s.model_id = r.str();
        s.producer = r.str();
        s.T = r.f32();
        s.Z = r.f32();
        if (out.version >= 2) {
            s.conv_kernel_size = r.pod<uint32_t>();
            s.conv_dim = r.pod<uint32_t>();
            s.conv_tail_rows = r.pod<uint32_t>();
            s.conv_tail = r.f32();
        }
        s.validate();
        out.summaries.push_back(std::move(s));
    }
    if (!r.done()) throw std::runtime_error("GDN runtime bundle has trailing payload fields");
    out.validate();
    return out;
}

std::vector<GdnRuntimeSelectedBlock> gdn_runtime_blocks(
    const GdnRuntimeBundle& bundle) {
    bundle.validate();
    const auto layers = group_layers(bundle);
    const auto& canonical = layers.begin()->second;
    std::vector<GdnRuntimeSelectedBlock> out;
    out.reserve(canonical.size());
    for (uint32_t i = 0; i < canonical.size(); ++i) {
        out.push_back({i, canonical[i]->position_start, canonical[i]->token_count});
    }
    return out;
}

GdnRuntimeReconstruction reconstruct_gdn_runtime_bundle_cpu(
    const GdnRuntimeBundle& bundle,
    const std::vector<uint32_t>& selected_block_ids,
    const KvMemCacheIdentity& expected_identity) {
    bundle.validate();
    require_identity(expected_identity);
    if (compare_cache_identity(bundle.identity, expected_identity) !=
        KvMemCacheCompatibility::ExactCompatible) {
        throw std::runtime_error("GDN runtime cache identity is not exactly compatible");
    }
    if (selected_block_ids.empty()) throw std::invalid_argument("GDN runtime selection is empty");

    const auto blocks = gdn_runtime_blocks(bundle);
    uint32_t previous = 0;
    bool first = true;
    for (uint32_t id : selected_block_ids) {
        if (id >= blocks.size()) throw std::out_of_range("GDN runtime block id out of range");
        if (!first && id <= previous) {
            throw std::invalid_argument("GDN runtime block ids must be strictly increasing");
        }
        previous = id;
        first = false;
    }

    bool contiguous = selected_block_ids.front() == 0;
    for (size_t i = 1; i < selected_block_ids.size(); ++i) {
        contiguous = contiguous && selected_block_ids[i] == selected_block_ids[i - 1] + 1;
    }
    const bool starts_zero = blocks[selected_block_ids.front()].position_start == 0;

    const auto layers = group_layers(bundle);
    GdnRuntimeReconstruction out;
    out.selected_block_ids = selected_block_ids;
    out.starts_at_zero = starts_zero;
    out.contiguous = contiguous;
    out.full_model_boundary_exact = starts_zero && contiguous;
    out.hidden_replay_required = !out.full_model_boundary_exact;
    out.seam_repair_required = out.hidden_replay_required;
    out.layers.reserve(layers.size());
    bool all_conv = true;

    for (const auto& [layer, all] : layers) {
        std::vector<GdnBlockAffineSummary> selected;
        selected.reserve(selected_block_ids.size());
        for (uint32_t id : selected_block_ids) selected.push_back(*all[id]);
        auto store = GdnSummaryStore::from_summaries(selected);
        store.require_model(bundle.model_id);
        std::vector<GdnSummarySpan> spans;
        spans.reserve(selected.size());
        for (const auto& sum : selected) spans.push_back({sum.position_start, sum.token_count});
        const auto r = store.reconstruct_zero_state(
            layer, spans,
            out.hidden_replay_required ? GdnSeamPolicy::PreparedInputApprox
                                       : GdnSeamPolicy::FullModelExact);

        GdnRuntimeLayerState ls;
        ls.layer_index = r.layer_index;
        ls.num_v_heads = r.num_v_heads;
        ls.head_dim = r.head_dim;
        ls.state_column_major = r.state_column_major;

        bool conv_ok = !selected.empty();
        uint32_t K = 0, C = 0;
        for (const auto& sum : selected) {
            if (sum.version < 3) { conv_ok = false; break; }
            if (K == 0) { K = sum.conv_kernel_size; C = sum.conv_dim; }
            if (sum.conv_kernel_size != K || sum.conv_dim != C) {
                throw std::runtime_error("GDN Conv1D geometry changes within one layer");
            }
        }
        if (conv_ok) {
            const uint32_t tail = K - 1;
            ls.conv_kernel_size = K;
            ls.conv_dim = C;
            ls.conv_state_channel_major.assign(static_cast<size_t>(C) * tail, 0.0f);
            for (const auto& sum : selected) {
                const uint32_t rows = sum.conv_tail_rows;
                if (sum.token_count >= tail) {
                    for (uint32_t c = 0; c < C; ++c) {
                        for (uint32_t rr = 0; rr < tail; ++rr) {
                            ls.conv_state_channel_major[static_cast<size_t>(c) * tail + rr] =
                                sum.conv_tail[static_cast<size_t>(rr) * C + c];
                        }
                    }
                } else {
                    const uint32_t shift = rows;
                    for (uint32_t c = 0; c < C; ++c) {
                        float* dst = ls.conv_state_channel_major.data() + static_cast<size_t>(c) * tail;
                        std::move(dst + shift, dst + tail, dst);
                        for (uint32_t rr = 0; rr < rows; ++rr) {
                            dst[tail - rows + rr] =
                                sum.conv_tail[static_cast<size_t>(rr) * C + c];
                        }
                    }
                }
            }
        } else {
            all_conv = false;
        }
        out.layers.push_back(std::move(ls));
    }
    out.conv_state_available = all_conv;
    if (out.full_model_boundary_exact && !out.conv_state_available) {
        // RC10.8 bundles can still reconstruct the recurrent matrix, but they
        // cannot claim an exact executable boundary without the Conv1D ring.
        out.full_model_boundary_exact = false;
        out.hidden_replay_required = true;
        out.seam_repair_required = true;
    }
    return out;
}

GdnSeamReplayPlan plan_gdn_seam_replay(
    const GdnRuntimeBundle& bundle,
    const std::vector<uint32_t>& selected_block_ids,
    uint32_t hidden_repair_tokens,
    double max_replay_fraction) {
    bundle.validate();
    if (selected_block_ids.empty()) throw std::invalid_argument("GDN seam plan selection is empty");
    if (!(max_replay_fraction > 0.0) || max_replay_fraction > 1.0 ||
        !std::isfinite(max_replay_fraction)) {
        throw std::invalid_argument("GDN seam max replay fraction must be in (0,1]");
    }
    const auto blocks = gdn_runtime_blocks(bundle);
    uint32_t previous = 0;
    bool first = true;
    for (uint32_t id : selected_block_ids) {
        if (id >= blocks.size()) throw std::out_of_range("GDN seam plan block id out of range");
        if (!first && id <= previous) throw std::invalid_argument("GDN seam plan block ids must be strictly increasing");
        previous = id;
        first = false;
    }

    uint32_t conv_repair = 0;
    for (const auto& s : bundle.summaries) {
        if (s.version >= 3 && s.conv_kernel_size > 0) {
            conv_repair = std::max(conv_repair, s.conv_kernel_size - 1);
        }
    }
    if (hidden_repair_tokens < conv_repair) {
        throw std::invalid_argument("hidden repair window is smaller than the Conv1D dependency width");
    }

    GdnSeamReplayPlan plan;
    plan.selected_block_ids = selected_block_ids;
    plan.conv_repair_tokens = conv_repair;
    plan.hidden_repair_tokens = hidden_repair_tokens;
    uint32_t compact = 0;
    std::vector<uint32_t> seam_starts;
    std::vector<uint32_t> seam_ids;
    std::vector<uint64_t> seam_sources;
    for (size_t i = 0; i < selected_block_ids.size(); ++i) {
        const auto& b = blocks[selected_block_ids[i]];
        const bool discontinuity =
            (i == 0) ? (b.position_start != 0) :
            (b.position_start != blocks[selected_block_ids[i-1]].position_start +
                                 blocks[selected_block_ids[i-1]].token_count);
        if (discontinuity) {
            ++plan.discontinuities;
            seam_starts.push_back(compact);
            seam_ids.push_back(selected_block_ids[i]);
            seam_sources.push_back(b.position_start);
        }
        plan.selected_tokens += b.token_count;
        compact += b.token_count;
    }
    plan.full_model_boundary_exact = plan.discontinuities == 0 &&
        blocks[selected_block_ids.front()].position_start == 0;
    if (plan.full_model_boundary_exact) {
        plan.parity_gate_required = false;
        plan.qualification_shortcut_eligible = true;
        return plan;
    }

    // Merge fixed-length replay windows in compact selected-context coordinates.
    uint32_t covered_end = 0;
    for (size_t i = 0; i < seam_starts.size(); ++i) {
        const uint32_t start = seam_starts[i];
        const uint32_t end = std::min<uint32_t>(plan.selected_tokens,
                                                start + hidden_repair_tokens);
        if (end <= covered_end) continue;
        const uint32_t effective_start = std::max(start, covered_end);
        GdnSeamReplayWindow w;
        w.downstream_block_id = seam_ids[i];
        w.compact_token_start = effective_start;
        w.token_count = end - effective_start;
        w.source_position_start = seam_sources[i] + (effective_start - start);
        plan.replay_tokens += w.token_count;
        plan.windows.push_back(w);
        covered_end = end;
    }
    plan.replay_fraction = plan.selected_tokens == 0 ? 1.0 :
        static_cast<double>(plan.replay_tokens) / static_cast<double>(plan.selected_tokens);
    plan.qualification_shortcut_eligible =
        plan.replay_tokens < plan.selected_tokens && plan.replay_fraction <= max_replay_fraction;
    plan.parity_gate_required = true;
    return plan;
}

} // namespace qw3
