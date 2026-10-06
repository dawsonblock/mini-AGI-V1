#pragma once

#include "kvmem_session_state.hpp"
#include "qw3/qw3.hpp"

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace qw3::detail {

// Registry for detachable logical KVMem sessions. Device/executor state is not
// stored here; `hot_slot` is only an affinity/mount descriptor maintained by
// the executor scheduler. Canonical host state is sufficient for deterministic
// cold rehydration on a compatible executor.
class KvMemSessionManager {
public:
    struct Record : KvMemSessionState {
        std::string id;
        std::string workspace_id;
        uint64_t version = 0;
        uint64_t created_at = 0;
        uint64_t last_access_at = 0;
        uint64_t cold_rehydrates = 0;
        int32_t hot_slot = -1;
    };

    KvMemSessionManager(uint32_t max_sessions = 64,
                        uint64_t host_token_limit = 8ull * 1024ull * 1024ull,
                        uint64_t host_byte_limit = 0)
        : max_sessions_(std::max<uint32_t>(1, max_sessions)),
          host_token_limit_(std::max<uint64_t>(1, host_token_limit)),
          host_byte_limit_(host_byte_limit) {}

    void configure(uint32_t max_sessions, uint64_t host_token_limit,
                   uint64_t host_byte_limit = 0) {
        std::lock_guard<std::mutex> lock(mu_);
        max_sessions_ = std::max<uint32_t>(1, max_sessions);
        host_token_limit_ = std::max<uint64_t>(1, host_token_limit);
        host_byte_limit_ = host_byte_limit;
        enforce_limits_locked();
    }

    Record start(const std::string &id, const std::string &workspace_id,
                 uint64_t now) {
        validate_id(id);
        std::lock_guard<std::mutex> lock(mu_);
        auto it = sessions_.find(id);
        if (it == sessions_.end() && sessions_.size() >= max_sessions_)
            throw std::runtime_error("KVMem logical session limit exceeded");

        std::string effective_workspace = workspace_id;
        if (it != sessions_.end()) {
            validate_workspace_locked(it->second, workspace_id);
            if (effective_workspace.empty()) effective_workspace = it->second.workspace_id;
            total_tokens_ -= it->second.tokens.size();
            total_host_bytes_ -= record_host_bytes(it->second);
        }

        Record next;
        next.id = id;
        next.workspace_id = effective_workspace;
        next.version = 1;
        next.created_at = now;
        next.last_access_at = now;
        next.hot_slot = -1;
        const uint64_t next_bytes = record_host_bytes(next);
        if (host_byte_limit_ > 0 && total_host_bytes_ + next_bytes > host_byte_limit_)
            throw std::runtime_error("KVMem logical-session host byte limit exceeded");
        sessions_.insert_or_assign(id, next);
        total_host_bytes_ += next_bytes;
        return next;
    }

    Record require(const std::string &id, const std::string &workspace_id,
                   uint64_t now) {
        validate_id(id);
        std::lock_guard<std::mutex> lock(mu_);
        auto it = sessions_.find(id);
        if (it == sessions_.end())
            throw std::runtime_error("KVMem logical session not found: " + id);
        validate_workspace_locked(it->second, workspace_id);
        it->second.last_access_at = now;
        return it->second;
    }

    // Trusted backend-only lookup for persistence/maintenance operations that
    // are already scoped to the local process. Never expose this path to an
    // untrusted request: caller-facing operations must use require() so a bound
    // workspace cannot be bypassed by omission.
    Record require_internal(const std::string &id, uint64_t now) {
        validate_id(id);
        std::lock_guard<std::mutex> lock(mu_);
        auto it = sessions_.find(id);
        if (it == sessions_.end())
            throw std::runtime_error("KVMem logical session not found: " + id);
        it->second.last_access_at = now;
        return it->second;
    }

    void mount(const std::string &id, uint32_t slot, bool cold_rehydrate,
               uint64_t now) {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = sessions_.find(id);
        if (it == sessions_.end())
            throw std::runtime_error("KVMem logical session not found: " + id);
        // One session per slot in the registry. A scheduler eviction must be
        // reflected here even if the caller forgot to unmount it first.
        for (auto &[other_id, record] : sessions_) {
            if (other_id != id && record.hot_slot == static_cast<int32_t>(slot))
                record.hot_slot = -1;
        }
        it->second.hot_slot = static_cast<int32_t>(slot);
        it->second.last_access_at = now;
        if (cold_rehydrate) ++it->second.cold_rehydrates;
    }

    void unmount(const std::string &id, uint64_t now) {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = sessions_.find(id);
        if (it == sessions_.end()) return;
        if (it->second.hot_slot >= 0) it->second.last_access_at = now;
        it->second.hot_slot = -1;
    }

    void unmount_all(uint64_t now) {
        std::lock_guard<std::mutex> lock(mu_);
        for (auto &[id, record] : sessions_) {
            (void)id;
            if (record.hot_slot >= 0) record.last_access_at = now;
            record.hot_slot = -1;
        }
    }

    void commit(const Record &record, uint64_t now, bool bump_version = true) {
        validate_id(record.id);
        std::lock_guard<std::mutex> lock(mu_);
        auto it = sessions_.find(record.id);
        if (it == sessions_.end())
            throw std::runtime_error("KVMem logical session disappeared: " + record.id);
        validate_workspace_locked(it->second, record.workspace_id);
        const uint64_t old_tokens = it->second.tokens.size();
        const uint64_t old_bytes = record_host_bytes(it->second);
        const uint64_t new_bytes = record_host_bytes(record);
        const uint64_t new_total = total_tokens_ - old_tokens + record.tokens.size();
        const uint64_t new_total_bytes = total_host_bytes_ - old_bytes + new_bytes;
        if (new_total > host_token_limit_)
            throw std::runtime_error("KVMem logical-session host token limit exceeded");
        if (host_byte_limit_ > 0 && new_total_bytes > host_byte_limit_)
            throw std::runtime_error("KVMem logical-session host byte limit exceeded");

        Record next = record;
        next.created_at = it->second.created_at;
        next.last_access_at = now;
        next.version = bump_version ? it->second.version + 1 : it->second.version;
        next.cold_rehydrates = it->second.cold_rehydrates;
        // Mount ownership is scheduler state and cannot be overwritten by a
        // stale detached Record copy.
        next.hot_slot = it->second.hot_slot;
        total_tokens_ = new_total;
        total_host_bytes_ = new_total_bytes;
        sessions_[record.id] = std::move(next);
    }

    KvMemSessionInfo info(const std::string &id) const {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = sessions_.find(id);
        if (it == sessions_.end()) return {};
        return to_info(it->second);
    }

    std::vector<KvMemSessionInfo> list() const {
        std::lock_guard<std::mutex> lock(mu_);
        std::vector<KvMemSessionInfo> out;
        out.reserve(sessions_.size());
        for (const auto &[id, rec] : sessions_) {
            (void)id;
            out.push_back(to_info(rec));
        }
        std::sort(out.begin(), out.end(), [](const auto &a, const auto &b) {
            if (a.hot != b.hot) return a.hot > b.hot;
            if (a.last_access_at != b.last_access_at)
                return a.last_access_at > b.last_access_at;
            return a.id < b.id;
        });
        return out;
    }

    bool erase(const std::string &id) {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = sessions_.find(id);
        if (it == sessions_.end()) return false;
        total_tokens_ -= it->second.tokens.size();
        total_host_bytes_ -= record_host_bytes(it->second);
        sessions_.erase(it);
        return true;
    }

    bool is_hot(const std::string &id) const {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = sessions_.find(id);
        return it != sessions_.end() && it->second.hot_slot >= 0;
    }

    int32_t hot_slot(const std::string &id) const {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = sessions_.find(id);
        return it == sessions_.end() ? -1 : it->second.hot_slot;
    }

    uint64_t total_tokens() const {
        std::lock_guard<std::mutex> lock(mu_);
        return total_tokens_;
    }
    uint64_t total_host_bytes() const {
        std::lock_guard<std::mutex> lock(mu_);
        return total_host_bytes_;
    }
    uint32_t max_sessions() const { return max_sessions_; }
    uint64_t host_token_limit() const { return host_token_limit_; }
    uint64_t host_byte_limit() const { return host_byte_limit_; }

    static uint64_t estimate_host_bytes(const Record &record) {
        return record_host_bytes(record);
    }

    // Import a validated durable canonical record. Restored state is always
    // cold: no executor-slot ownership survives persistence.
    void import_cold(Record record, uint64_t now) {
        validate_id(record.id);
        std::lock_guard<std::mutex> lock(mu_);
        auto it = sessions_.find(record.id);
        if (it == sessions_.end() && sessions_.size() >= max_sessions_)
            throw std::runtime_error("KVMem logical session limit exceeded");
        if (it != sessions_.end()) validate_workspace_locked(it->second, record.workspace_id);
        const uint64_t old_tokens = it == sessions_.end() ? 0 : it->second.tokens.size();
        const uint64_t old_bytes = it == sessions_.end() ? 0 : record_host_bytes(it->second);
        const uint64_t new_bytes = record_host_bytes(record);
        const uint64_t new_total_tokens = total_tokens_ - old_tokens + record.tokens.size();
        const uint64_t new_total_bytes = total_host_bytes_ - old_bytes + new_bytes;
        if (new_total_tokens > host_token_limit_)
            throw std::runtime_error("KVMem logical-session host token limit exceeded");
        if (host_byte_limit_ > 0 && new_total_bytes > host_byte_limit_)
            throw std::runtime_error("KVMem logical-session host byte limit exceeded");
        record.hot_slot = -1;
        record.last_access_at = now;
        sessions_.insert_or_assign(record.id, std::move(record));
        total_tokens_ = new_total_tokens;
        total_host_bytes_ = new_total_bytes;
    }

private:
    static void validate_id(const std::string &id) {
        if (id.empty() || id.size() > 128)
            throw std::invalid_argument("KVMem session id must be 1..128 bytes");
        const bool valid = std::all_of(id.begin(), id.end(), [](unsigned char c) {
            return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                   (c >= '0' && c <= '9') || c == '-' || c == '_' ||
                   c == '.' || c == ':';
        });
        if (!valid)
            throw std::invalid_argument("KVMem session id contains unsupported characters");
    }

    static void validate_workspace_locked(const Record &record,
                                          const std::string &workspace_id) {
        if (!record.workspace_id.empty() && record.workspace_id != workspace_id)
            throw std::runtime_error(
                "KVMem session workspace binding mismatch for " + record.id);
    }

    static KvMemSessionInfo to_info(const Record &record) {
        KvMemSessionInfo out;
        out.found = true;
        out.id = record.id;
        out.workspace_id = record.workspace_id;
        out.status = record.hot_slot >= 0 ? "hot" : "cold";
        out.version = record.version;
        out.token_count = record.tokens.size();
        out.created_at = record.created_at;
        out.last_access_at = record.last_access_at;
        out.cold_rehydrates = record.cold_rehydrates;
        out.host_bytes = record_host_bytes(record);
        out.hot = record.hot_slot >= 0;
        out.executor_slot = record.hot_slot;
        return out;
    }

    static uint64_t record_host_bytes(const Record &record) {
        return static_cast<uint64_t>(record.id.size() + record.workspace_id.size()) +
               record.estimated_host_bytes();
    }

    void enforce_limits_locked() const {
        if (sessions_.size() > max_sessions_ || total_tokens_ > host_token_limit_ ||
            (host_byte_limit_ > 0 && total_host_bytes_ > host_byte_limit_))
            throw std::runtime_error(
                "new KVMem session limits are below current registry usage");
    }

    mutable std::mutex mu_;
    std::unordered_map<std::string, Record> sessions_;
    uint64_t total_tokens_ = 0;
    uint64_t total_host_bytes_ = 0;
    uint32_t max_sessions_;
    uint64_t host_token_limit_;
    uint64_t host_byte_limit_;
};

} // namespace qw3::detail
