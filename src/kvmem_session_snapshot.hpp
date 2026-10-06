#pragma once

#include "kvmem_session_manager.hpp"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace qw3::detail {

// Durable canonical-session snapshot store. These files are derived cache
// artifacts only: they contain enough host-side request state to cold-rehydrate
// the executor after restart, but never represent authorization or the source
// of truth for an agent trajectory.
class KvMemSessionSnapshotStore {
public:
    explicit KvMemSessionSnapshotStore(std::string directory = {},
                                       std::string runtime_fingerprint = {})
        : directory_(std::move(directory)),
          runtime_fingerprint_(std::move(runtime_fingerprint)) {}

    void configure(std::string directory, std::string runtime_fingerprint) {
        directory_ = std::move(directory);
        runtime_fingerprint_ = std::move(runtime_fingerprint);
    }

    bool enabled() const { return !directory_.empty(); }
    const std::string &directory() const { return directory_; }
    const std::string &runtime_fingerprint() const { return runtime_fingerprint_; }

    KvMemSessionSnapshotInfo save(
            const KvMemSessionManager::Record &record) const {
        require_enabled();
        validate_persistable(record);
        std::filesystem::create_directories(directory_);

        std::vector<uint8_t> payload;
        append_string(payload, record.id);
        append_string(payload, record.workspace_id);
        append_string(payload, runtime_fingerprint_);
        append_u64(payload, record.version);
        append_u64(payload, record.created_at);
        const uint64_t saved_at = unix_now();
        append_u64(payload, saved_at);
        append_u64(payload, record.cold_rehydrates);
        append_u64(payload, record.input_embedding_fingerprint);

        append_u64(payload, record.tokens.size());
        for (uint32_t token : record.tokens) append_u32(payload, token);

        append_u64(payload, record.input_embedding_overrides.size());
        for (const auto &item : record.input_embedding_overrides) {
            append_u32(payload, item.token_id);
            append_u32(payload, item.source_token_id);
            for (uint32_t pos : item.position) append_u32(payload, pos);
            append_u32(payload, item.embedding_row);
            append_u64(payload, item.embedding.size());
            for (float v : item.embedding) append_float(payload, v);
        }

        append_u64(payload, record.input_mrope_positions.size());
        for (const auto &pos : record.input_mrope_positions)
            for (uint32_t v : pos) append_u32(payload, v);

        append_u64(payload, record.pinned_token_spans.size());
        for (const auto &span : record.pinned_token_spans) {
            append_u32(payload, span.begin);
            append_u32(payload, span.end);
            append_u32(payload, static_cast<uint32_t>(span.reason));
        }

        const uint64_t checksum = fnv1a64(payload.data(), payload.size());
        const auto final_path = path_for(record.id);
        auto temp_path = final_path;
        temp_path += ".tmp";
        {
            std::ofstream out(temp_path, std::ios::binary | std::ios::trunc);
            if (!out) throw std::runtime_error("cannot create KVMem session snapshot");
            const char magic[8] = {'Q','W','3','K','V','S','4','\0'};
            out.write(magic, sizeof(magic));
            write_u32(out, kFormatVersion);
            write_u64(out, payload.size());
            write_u64(out, checksum);
            if (!payload.empty()) {
                out.write(reinterpret_cast<const char *>(payload.data()),
                          static_cast<std::streamsize>(payload.size()));
            }
            out.flush();
            if (!out) throw std::runtime_error("failed writing KVMem session snapshot");
        }
        std::error_code ec;
        std::filesystem::rename(temp_path, final_path, ec);
        if (ec) {
            // Windows-style replacement is not relevant to supported Linux
            // runtime but this also makes repeated host tests deterministic.
            std::filesystem::remove(final_path, ec);
            ec.clear();
            std::filesystem::rename(temp_path, final_path, ec);
            if (ec) {
                std::filesystem::remove(temp_path);
                throw std::runtime_error("failed publishing KVMem session snapshot: " +
                                         ec.message());
            }
        }
        return make_info(record, final_path.string(), saved_at);
    }

    KvMemSessionManager::Record load(const std::string &id,
                                     KvMemSessionSnapshotInfo *info = nullptr) const {
        require_enabled();
        const auto path = path_for(id);
        std::ifstream in(path, std::ios::binary);
        if (!in) throw std::runtime_error("KVMem session snapshot not found: " + id);

        char magic[8]{};
        in.read(magic, sizeof(magic));
        const char expected[8] = {'Q','W','3','K','V','S','4','\0'};
        if (!in || std::memcmp(magic, expected, sizeof(expected)) != 0)
            throw std::runtime_error("invalid KVMem session snapshot magic");
        const uint32_t version = read_u32(in);
        if (version != kFormatVersion)
            throw std::runtime_error("unsupported KVMem session snapshot version");
        const uint64_t payload_size = read_u64(in);
        const uint64_t checksum = read_u64(in);
        if (payload_size > kMaxSnapshotBytes)
            throw std::runtime_error("KVMem session snapshot exceeds safety limit");
        std::vector<uint8_t> payload(static_cast<size_t>(payload_size));
        if (payload_size) {
            in.read(reinterpret_cast<char *>(payload.data()),
                    static_cast<std::streamsize>(payload_size));
        }
        if (!in || fnv1a64(payload.data(), payload.size()) != checksum)
            throw std::runtime_error("KVMem session snapshot checksum mismatch");

        Reader r(payload);
        KvMemSessionManager::Record record;
        record.id = r.string();
        record.workspace_id = r.string();
        const std::string stored_fingerprint = r.string();
        if (record.id != id)
            throw std::runtime_error("KVMem session snapshot ID mismatch");
        if (stored_fingerprint != runtime_fingerprint_)
            throw std::runtime_error(
                "KVMem session snapshot runtime fingerprint mismatch");
        record.version = r.u64();
        record.created_at = r.u64();
        const uint64_t saved_at = r.u64();
        record.cold_rehydrates = r.u64();
        record.input_embedding_fingerprint = r.u64();
        record.hot_slot = -1;

        const uint64_t token_count = r.count("tokens");
        record.tokens.reserve(static_cast<size_t>(token_count));
        for (uint64_t i = 0; i < token_count; ++i)
            record.tokens.push_back(r.u32());

        const uint64_t override_count = r.count("embedding overrides", 1u << 20);
        record.input_embedding_overrides.reserve(static_cast<size_t>(override_count));
        for (uint64_t i = 0; i < override_count; ++i) {
            GenerationOptions::InputEmbeddingOverride item;
            item.token_id = r.u32();
            item.source_token_id = r.u32();
            for (auto &v : item.position) v = r.u32();
            item.embedding_row = r.u32();
            const uint64_t floats = r.count("embedding floats", 1ull << 30);
            item.embedding.reserve(static_cast<size_t>(floats));
            for (uint64_t j = 0; j < floats; ++j) item.embedding.push_back(r.f32());
            record.input_embedding_overrides.push_back(std::move(item));
        }

        const uint64_t mrope_count = r.count("M-RoPE rows");
        record.input_mrope_positions.resize(static_cast<size_t>(mrope_count));
        for (auto &pos : record.input_mrope_positions)
            for (auto &v : pos) v = r.u32();

        const uint64_t pin_count = r.count("pinned spans", 1u << 24);
        record.pinned_token_spans.reserve(static_cast<size_t>(pin_count));
        for (uint64_t i = 0; i < pin_count; ++i) {
            GenerationOptions::KvMemPinnedTokenSpan span;
            span.begin = r.u32();
            span.end = r.u32();
            const uint32_t reason = r.u32();
            if (reason > static_cast<uint32_t>(
                             GenerationOptions::KvMemPinnedReason::RootTask))
                throw std::runtime_error("invalid KVMem snapshot pinned-span reason");
            span.reason = static_cast<GenerationOptions::KvMemPinnedReason>(reason);
            record.pinned_token_spans.push_back(span);
        }
        r.require_eof();

        if (info) *info = make_info(record, path.string(), saved_at);
        return record;
    }

    KvMemSessionSnapshotInfo info(const std::string &id) const {
        if (!enabled()) return {};
        const auto path = path_for(id);
        if (!std::filesystem::exists(path)) return {};
        KvMemSessionSnapshotInfo out;
        (void)load(id, &out); // validates integrity + runtime compatibility
        return out;
    }

private:
    static constexpr uint32_t kFormatVersion = 1;
    static constexpr uint64_t kMaxSnapshotBytes = 64ull * 1024ull * 1024ull * 1024ull;

    class Reader {
    public:
        explicit Reader(const std::vector<uint8_t> &data) : data_(data) {}
        uint32_t u32() { return pod<uint32_t>(); }
        uint64_t u64() { return pod<uint64_t>(); }
        float f32() { return pod<float>(); }
        std::string string() {
            const uint64_t n = count("string bytes", 16u * 1024u * 1024u);
            need(n);
            std::string value(reinterpret_cast<const char *>(data_.data() + off_),
                              static_cast<size_t>(n));
            off_ += static_cast<size_t>(n);
            return value;
        }
        uint64_t count(const char *what,
                       uint64_t max = static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
            const uint64_t n = u64();
            if (n > max) throw std::runtime_error(std::string("invalid snapshot ") + what);
            return n;
        }
        void require_eof() const {
            if (off_ != data_.size())
                throw std::runtime_error("KVMem snapshot has trailing payload bytes");
        }
    private:
        template <typename T> T pod() {
            static_assert(std::is_trivially_copyable_v<T>);
            need(sizeof(T));
            T value{};
            std::memcpy(&value, data_.data() + off_, sizeof(T));
            off_ += sizeof(T);
            return value;
        }
        void need(uint64_t n) const {
            if (n > data_.size() - off_)
                throw std::runtime_error("truncated KVMem session snapshot");
        }
        const std::vector<uint8_t> &data_;
        size_t off_ = 0;
    };

    static uint64_t unix_now() {
        return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    }
    static uint64_t fnv1a64(const uint8_t *data, size_t size) {
        uint64_t h = 1469598103934665603ULL;
        for (size_t i = 0; i < size; ++i) {
            h ^= static_cast<uint64_t>(data[i]);
            h *= 1099511628211ULL;
        }
        return h;
    }
    static void append_u32(std::vector<uint8_t> &out, uint32_t v) { append_pod(out, v); }
    static void append_u64(std::vector<uint8_t> &out, uint64_t v) { append_pod(out, v); }
    static void append_float(std::vector<uint8_t> &out, float v) { append_pod(out, v); }
    template <typename T> static void append_pod(std::vector<uint8_t> &out, T v) {
        static_assert(std::is_trivially_copyable_v<T>);
        const auto *p = reinterpret_cast<const uint8_t *>(&v);
        out.insert(out.end(), p, p + sizeof(T));
    }
    static void append_string(std::vector<uint8_t> &out, const std::string &v) {
        append_u64(out, v.size());
        out.insert(out.end(), v.begin(), v.end());
    }
    static void write_u32(std::ostream &out, uint32_t v) {
        out.write(reinterpret_cast<const char *>(&v), sizeof(v));
    }
    static void write_u64(std::ostream &out, uint64_t v) {
        out.write(reinterpret_cast<const char *>(&v), sizeof(v));
    }
    static uint32_t read_u32(std::istream &in) {
        uint32_t v{}; in.read(reinterpret_cast<char *>(&v), sizeof(v));
        if (!in) throw std::runtime_error("truncated KVMem session snapshot header");
        return v;
    }
    static uint64_t read_u64(std::istream &in) {
        uint64_t v{}; in.read(reinterpret_cast<char *>(&v), sizeof(v));
        if (!in) throw std::runtime_error("truncated KVMem session snapshot header");
        return v;
    }
    static void validate_persistable(const KvMemSessionManager::Record &record) {
        for (const auto &item : record.input_embedding_overrides) {
            if (item.storage) {
                throw std::runtime_error(
                    "KVMem session snapshot cannot persist device-only input embedding storage; "
                    "rebuild that session from the authoritative workspace ledger");
            }
        }
    }
    void require_enabled() const {
        if (!enabled())
            throw std::runtime_error("KVMem session snapshot directory is not configured");
    }
    std::filesystem::path path_for(const std::string &id) const {
        // KvMemSessionManager already constrains IDs; repeat enough validation
        // here to prevent path traversal when info/load is called directly.
        if (id.empty() || id.size() > 128 ||
            id.find('/') != std::string::npos || id.find('\\') != std::string::npos ||
            id == "." || id == "..")
            throw std::invalid_argument("invalid KVMem session snapshot ID");
        return std::filesystem::path(directory_) / (id + ".qkvs");
    }
    KvMemSessionSnapshotInfo make_info(
            const KvMemSessionManager::Record &record,
            const std::string &path, uint64_t saved_at) const {
        KvMemSessionSnapshotInfo out;
        out.found = true;
        out.id = record.id;
        out.workspace_id = record.workspace_id;
        out.path = path;
        out.runtime_fingerprint = runtime_fingerprint_;
        out.version = record.version;
        out.token_count = record.tokens.size();
        out.host_bytes = KvMemSessionManager::estimate_host_bytes(record);
        out.created_at = record.created_at;
        out.saved_at = saved_at;
        return out;
    }

    std::string directory_;
    std::string runtime_fingerprint_;
};

} // namespace qw3::detail
