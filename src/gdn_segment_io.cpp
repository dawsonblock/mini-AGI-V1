#include "qw3/gdn_segment_io.hpp"
#include "qw3/gdn_reference.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace qw3 {
namespace {

constexpr char kCaptureMagic[8] = {'Q','G','D','N','C','A','P','1'};
constexpr char kSummaryMagic[8] = {'Q','G','D','N','S','U','M','1'};
constexpr uint32_t kMaxHeads = 1024;
constexpr uint32_t kMaxHeadDim = 256;
constexpr uint32_t kMaxTokens = 1u << 24;
constexpr uint64_t kMaxElements = uint64_t{1} << 31;
constexpr uint32_t kMaxString = 1u << 20;

uint64_t fnv1a64(const uint8_t* data, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) {
        h ^= data[i];
        h *= 1099511628211ull;
    }
    return h;
}

class ByteWriter {
public:
    template <class T> void pod(T v) {
        static_assert(std::is_trivially_copyable_v<T>);
        const auto* p = reinterpret_cast<const uint8_t*>(&v);
        bytes.insert(bytes.end(), p, p + sizeof(T));
    }
    void str(const std::string& s) {
        if (s.size() > kMaxString) throw std::invalid_argument("GDN string too large");
        pod<uint32_t>(static_cast<uint32_t>(s.size()));
        bytes.insert(bytes.end(), s.begin(), s.end());
    }
    void f32(const std::vector<float>& v) {
        if (v.size() > kMaxElements) throw std::invalid_argument("GDN vector too large");
        pod<uint64_t>(static_cast<uint64_t>(v.size()));
        const auto* p = reinterpret_cast<const uint8_t*>(v.data());
        bytes.insert(bytes.end(), p, p + v.size() * sizeof(float));
    }
    std::vector<uint8_t> bytes;
};

class ByteReader {
public:
    explicit ByteReader(std::vector<uint8_t> data) : bytes(std::move(data)) {}
    template <class T> T pod() {
        static_assert(std::is_trivially_copyable_v<T>);
        require(sizeof(T));
        T v{};
        std::memcpy(&v, bytes.data() + off, sizeof(T));
        off += sizeof(T);
        return v;
    }
    std::string str() {
        const uint32_t n = pod<uint32_t>();
        if (n > kMaxString) throw std::runtime_error("GDN string length rejected");
        require(n);
        std::string s(reinterpret_cast<const char*>(bytes.data() + off), n);
        off += n;
        return s;
    }
    std::vector<float> f32() {
        const uint64_t n = pod<uint64_t>();
        if (n > kMaxElements || n > std::numeric_limits<size_t>::max() / sizeof(float)) {
            throw std::runtime_error("GDN vector length rejected");
        }
        const size_t count = static_cast<size_t>(n);
        require(count * sizeof(float));
        std::vector<float> out(count);
        if (count) std::memcpy(out.data(), bytes.data() + off, count * sizeof(float));
        off += count * sizeof(float);
        return out;
    }
    bool done() const { return off == bytes.size(); }
private:
    void require(size_t n) const {
        if (n > bytes.size() - off) throw std::runtime_error("truncated GDN payload");
    }
    std::vector<uint8_t> bytes;
    size_t off = 0;
};

void validate_finite(const std::vector<float>& v, const char* name) {
    for (float x : v) if (!std::isfinite(x)) {
        throw std::invalid_argument(std::string(name) + " contains non-finite value");
    }
}

void write_envelope(const std::string& path, const char magic[8], const std::vector<uint8_t>& payload) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot open GDN output: " + path);
    out.write(magic, 8);
    const uint64_t n = payload.size();
    const uint64_t hash = fnv1a64(payload.data(), payload.size());
    out.write(reinterpret_cast<const char*>(&n), sizeof(n));
    out.write(reinterpret_cast<const char*>(&hash), sizeof(hash));
    if (n) out.write(reinterpret_cast<const char*>(payload.data()), static_cast<std::streamsize>(n));
    if (!out) throw std::runtime_error("failed writing GDN output: " + path);
}

std::vector<uint8_t> read_envelope(const std::string& path, const char magic[8]) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open GDN input: " + path);
    char got[8]{};
    uint64_t n = 0, expected = 0;
    in.read(got, 8);
    in.read(reinterpret_cast<char*>(&n), sizeof(n));
    in.read(reinterpret_cast<char*>(&expected), sizeof(expected));
    if (!in || std::memcmp(got, magic, 8) != 0) throw std::runtime_error("invalid GDN file header");
    constexpr uint64_t kMaxFilePayload = uint64_t{8} << 30;
    if (n > kMaxFilePayload || n > std::numeric_limits<size_t>::max()) {
        throw std::runtime_error("GDN payload length rejected");
    }
    std::vector<uint8_t> payload(static_cast<size_t>(n));
    if (n) in.read(reinterpret_cast<char*>(payload.data()), static_cast<std::streamsize>(n));
    if (!in) throw std::runtime_error("truncated GDN file payload");
    char trailing = 0;
    if (in.read(&trailing, 1)) throw std::runtime_error("GDN file has trailing bytes");
    if (fnv1a64(payload.data(), payload.size()) != expected) throw std::runtime_error("GDN payload checksum mismatch");
    return payload;
}

size_t matrix_elems(uint32_t heads, uint32_t dim) {
    if (!heads || heads > kMaxHeads || !dim || dim > kMaxHeadDim) {
        throw std::invalid_argument("GDN summary dimensions rejected");
    }
    const uint64_t n = static_cast<uint64_t>(heads) * dim * dim;
    if (n > kMaxElements) throw std::invalid_argument("GDN summary size rejected");
    return static_cast<size_t>(n);
}

} // namespace

void GdnPreparedCapture::validate() const {
    if (version != 1 && version != 2) throw std::invalid_argument("unsupported GDN capture version");
    if (!T || T > kMaxTokens || !num_k_heads || !num_v_heads ||
        num_k_heads > kMaxHeads || num_v_heads > kMaxHeads ||
        !head_dim || head_dim > kMaxHeadDim || num_v_heads % num_k_heads != 0) {
        throw std::invalid_argument("invalid GDN capture dimensions");
    }
    const uint64_t kd = static_cast<uint64_t>(T) * num_k_heads * head_dim;
    const uint64_t vd = static_cast<uint64_t>(T) * num_v_heads * head_dim;
    const uint64_t gd = static_cast<uint64_t>(T) * num_v_heads;
    if (kd > kMaxElements || vd > kMaxElements || gd > kMaxElements ||
        q.size() != kd || k.size() != kd || v.size() != vd ||
        decay.size() != gd || beta.size() != gd) {
        throw std::invalid_argument("GDN capture shape mismatch");
    }
    validate_finite(q, "q"); validate_finite(k, "k"); validate_finite(v, "v");
    validate_finite(decay, "decay"); validate_finite(beta, "beta");
    if (version >= 2) {
        if (conv_kernel_size < 2 || conv_kernel_size > 64 || conv_dim == 0 ||
            static_cast<uint64_t>(T) * conv_dim > kMaxElements ||
            preconv.size() != static_cast<size_t>(T) * conv_dim) {
            throw std::invalid_argument("GDN capture Conv1D metadata/shape mismatch");
        }
        validate_finite(preconv, "preconv");
    } else if (conv_kernel_size != 0 || conv_dim != 0 || !preconv.empty()) {
        throw std::invalid_argument("GDN v1 capture cannot carry Conv1D fields");
    }
    for (float g : decay) if (g < 0.0f || g > 1.0001f) throw std::invalid_argument("GDN decay out of range");
    for (float b : beta) if (b < 0.0f || b > 1.0f) throw std::invalid_argument("GDN beta out of range");
}

void GdnBlockAffineSummary::validate() const {
    if ((version != 1 && version != 2 && version != 3) || token_count == 0) throw std::invalid_argument("invalid GDN summary metadata");
    if (version >= 2 && model_id.empty()) throw std::invalid_argument("GDN v2+ summary requires model_id");
    const size_t n = matrix_elems(num_v_heads, head_dim);
    if (T.size() != n || Z.size() != n) throw std::invalid_argument("GDN summary shape mismatch");
    validate_finite(T, "T"); validate_finite(Z, "Z");
    if (version >= 3) {
        if (conv_kernel_size < 2 || conv_kernel_size > 64 || conv_dim == 0) {
            throw std::invalid_argument("GDN v3 summary Conv1D metadata rejected");
        }
        const uint32_t expected_rows = std::min<uint32_t>(token_count, conv_kernel_size - 1);
        if (conv_tail_rows != expected_rows ||
            conv_tail.size() != static_cast<size_t>(conv_tail_rows) * conv_dim) {
            throw std::invalid_argument("GDN v3 summary Conv1D tail shape mismatch");
        }
        validate_finite(conv_tail, "conv_tail");
    } else if (conv_kernel_size != 0 || conv_dim != 0 || conv_tail_rows != 0 || !conv_tail.empty()) {
        throw std::invalid_argument("legacy GDN summary cannot carry Conv1D tail fields");
    }
}

void write_gdn_capture(const std::string& path, const GdnPreparedCapture& c) {
    c.validate();
    ByteWriter w;
    w.pod<uint32_t>(c.version); w.pod<uint32_t>(c.layer_index); w.pod<uint64_t>(c.position_start);
    w.pod<uint32_t>(c.T); w.pod<uint32_t>(c.num_k_heads); w.pod<uint32_t>(c.num_v_heads); w.pod<uint32_t>(c.head_dim);
    w.str(c.model_id); w.f32(c.q); w.f32(c.k); w.f32(c.v); w.f32(c.decay); w.f32(c.beta);
    if (c.version >= 2) {
        w.pod<uint32_t>(c.conv_kernel_size);
        w.pod<uint32_t>(c.conv_dim);
        w.f32(c.preconv);
    }
    write_envelope(path, kCaptureMagic, w.bytes);
}

GdnPreparedCapture read_gdn_capture(const std::string& path) {
    ByteReader r(read_envelope(path, kCaptureMagic));
    GdnPreparedCapture c;
    c.version = r.pod<uint32_t>(); c.layer_index = r.pod<uint32_t>(); c.position_start = r.pod<uint64_t>();
    c.T = r.pod<uint32_t>(); c.num_k_heads = r.pod<uint32_t>(); c.num_v_heads = r.pod<uint32_t>(); c.head_dim = r.pod<uint32_t>();
    c.model_id = r.str(); c.q = r.f32(); c.k = r.f32(); c.v = r.f32(); c.decay = r.f32(); c.beta = r.f32();
    if (c.version >= 2) {
        c.conv_kernel_size = r.pod<uint32_t>();
        c.conv_dim = r.pod<uint32_t>();
        c.preconv = r.f32();
    }
    if (!r.done()) throw std::runtime_error("GDN capture has trailing payload fields");
    c.validate();
    return c;
}

void write_gdn_summary_archive(const std::string& path, const std::vector<GdnBlockAffineSummary>& summaries) {
    if (summaries.empty()) throw std::invalid_argument("cannot write empty GDN summary archive");
    const uint32_t archive_version = summaries.front().version;
    if (archive_version < 1 || archive_version > 3) {
        throw std::invalid_argument("unsupported GDN summary archive generation");
    }
    ByteWriter w;
    w.pod<uint32_t>(archive_version);
    w.pod<uint32_t>(static_cast<uint32_t>(summaries.size()));
    for (const auto& s : summaries) {
        s.validate();
        if (s.version != archive_version) {
            throw std::invalid_argument("GDN summary archive cannot mix block generations");
        }
        w.pod<uint32_t>(s.version); w.pod<uint32_t>(s.layer_index); w.pod<uint64_t>(s.position_start);
        w.pod<uint32_t>(s.token_count); w.pod<uint32_t>(s.num_v_heads); w.pod<uint32_t>(s.head_dim);
        if (archive_version >= 2) w.str(s.model_id);
        w.str(s.producer); w.f32(s.T); w.f32(s.Z);
        if (archive_version >= 3) {
            w.pod<uint32_t>(s.conv_kernel_size); w.pod<uint32_t>(s.conv_dim);
            w.pod<uint32_t>(s.conv_tail_rows); w.f32(s.conv_tail);
        }
    }
    write_envelope(path, kSummaryMagic, w.bytes);
}

std::vector<GdnBlockAffineSummary> read_gdn_summary_archive(const std::string& path) {
    ByteReader r(read_envelope(path, kSummaryMagic));
    const uint32_t version = r.pod<uint32_t>();
    const uint32_t count = r.pod<uint32_t>();
    if ((version != 1 && version != 2 && version != 3) || count == 0 || count > (1u << 20)) {
        throw std::runtime_error("GDN summary archive metadata rejected");
    }
    std::vector<GdnBlockAffineSummary> out;
    out.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        GdnBlockAffineSummary s;
        s.version = r.pod<uint32_t>(); s.layer_index = r.pod<uint32_t>(); s.position_start = r.pod<uint64_t>();
        s.token_count = r.pod<uint32_t>(); s.num_v_heads = r.pod<uint32_t>(); s.head_dim = r.pod<uint32_t>();
        if (version >= 2) s.model_id = r.str();
        s.producer = r.str(); s.T = r.f32(); s.Z = r.f32();
        if (version >= 3) {
            s.conv_kernel_size = r.pod<uint32_t>(); s.conv_dim = r.pod<uint32_t>();
            s.conv_tail_rows = r.pod<uint32_t>(); s.conv_tail = r.f32();
        }
        if ((version == 1 && s.version != 1) || (version == 2 && s.version != 2) ||
            (version == 3 && s.version != 3)) {
            throw std::runtime_error("GDN summary archive/block version mismatch");
        }
        s.validate(); out.push_back(std::move(s));
    }
    if (!r.done()) throw std::runtime_error("GDN summary archive has trailing payload fields");
    return out;
}

std::vector<GdnBlockAffineSummary> gdn_summarize_capture_reference(const GdnPreparedCapture& capture, uint32_t block_tokens) {
    capture.validate();
    if (block_tokens == 0) throw std::invalid_argument("GDN block_tokens must be non-zero");
    std::vector<GdnBlockAffineSummary> out;
    const uint32_t D = capture.head_dim;
    for (uint32_t begin = 0; begin < capture.T; begin += block_tokens) {
        const uint32_t count = std::min(block_tokens, capture.T - begin);
        GdnBlockAffineSummary block;
        block.version = capture.version >= 2 ? 3u : 2u;
        block.layer_index = capture.layer_index;
        block.position_start = capture.position_start + begin;
        block.token_count = count;
        block.num_v_heads = capture.num_v_heads;
        block.head_dim = D;
        block.model_id = capture.model_id;
        block.producer = "fp64-reference";
        if (capture.version >= 2) {
            block.conv_kernel_size = capture.conv_kernel_size;
            block.conv_dim = capture.conv_dim;
            block.conv_tail_rows = std::min<uint32_t>(count, capture.conv_kernel_size - 1);
            const uint32_t tail_begin = begin + count - block.conv_tail_rows;
            block.conv_tail.resize(static_cast<size_t>(block.conv_tail_rows) * block.conv_dim);
            for (uint32_t r = 0; r < block.conv_tail_rows; ++r) {
                const float* src = capture.preconv.data() +
                    static_cast<size_t>(tail_begin + r) * block.conv_dim;
                std::copy_n(src, block.conv_dim,
                            block.conv_tail.data() + static_cast<size_t>(r) * block.conv_dim);
            }
        }
        const size_t per = static_cast<size_t>(D) * D;
        block.T.resize(static_cast<size_t>(capture.num_v_heads) * per);
        block.Z.resize(static_cast<size_t>(capture.num_v_heads) * per);
        for (uint32_t vh = 0; vh < capture.num_v_heads; ++vh) {
            const uint32_t kh = vh % capture.num_k_heads;
            std::vector<GdnReferenceToken> tokens;
            tokens.reserve(count);
            for (uint32_t ti = 0; ti < count; ++ti) {
                const uint32_t t = begin + ti;
                GdnReferenceToken tok;
                tok.k.resize(D); tok.v.resize(D);
                const size_t kb = (static_cast<size_t>(t) * capture.num_k_heads + kh) * D;
                const size_t vb = (static_cast<size_t>(t) * capture.num_v_heads + vh) * D;
                for (uint32_t d = 0; d < D; ++d) { tok.k[d] = capture.k[kb + d]; tok.v[d] = capture.v[vb + d]; }
                tok.g = capture.decay[static_cast<size_t>(t) * capture.num_v_heads + vh];
                tok.beta = capture.beta[static_cast<size_t>(t) * capture.num_v_heads + vh];
                tokens.push_back(std::move(tok));
            }
            const auto s = gdn_compose_tokens(tokens);
            for (size_t i = 0; i < per; ++i) {
                block.T[static_cast<size_t>(vh) * per + i] = static_cast<float>(s.T[i]);
                block.Z[static_cast<size_t>(vh) * per + i] = static_cast<float>(s.Z[i]);
            }
        }
        block.validate(); out.push_back(std::move(block));
    }
    return out;
}

} // namespace qw3
