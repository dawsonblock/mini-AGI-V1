#pragma once

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace qw3::detail {

// Host-only affinity scheduler for detachable logical KVMem sessions.
//
// The scheduler knows nothing about CUDA. A slot means "one independently
// mutable executor lineage". The current QW3 native backend certifies exactly
// one physical slot, but this implementation is N-slot capable so a future
// executor pool can adopt it without changing the session ABI or API surface.
class KvMemExecutorScheduler {
public:
    struct Slot {
        uint32_t index = 0;
        std::string session_id;
        bool busy = false;
        uint64_t generation = 0;
        uint64_t last_used_at = 0;
        uint64_t mounts = 0;
        uint64_t cold_mounts = 0;
        // A failed execution dirties the mutable executor lineage. The next
        // lease may reuse the physical slot only as a cold mount and must
        // rebuild/reset state before treating it as warm again.
        bool dirty = false;
        uint64_t fault_count = 0;
        uint64_t reset_count = 0;
        // Exact active lease fencing token. Zero means idle. A stale lease for
        // the same session must never be able to release a newer owner.
        uint64_t active_lease_id = 0;
    };

    struct Lease {
        uint32_t slot = 0;
        uint64_t lease_id = 0;
        std::string session_id;
        std::string evicted_session_id;
        bool warm_hit = false;
        bool cold_mount = false;
        bool requires_cold_reset = false;
        explicit operator bool() const { return lease_id != 0; }
    };

    struct Info {
        uint32_t slot_count = 0;
        uint32_t busy_slots = 0;
        uint32_t mounted_sessions = 0;
        uint64_t lease_sequence = 0;
        uint64_t warm_hits = 0;
        uint64_t cold_mounts = 0;
        uint64_t backpressure_rejections = 0;
        uint64_t faulted_releases = 0;
        uint64_t forced_cold_resets = 0;
        std::vector<Slot> slots;
    };

    explicit KvMemExecutorScheduler(uint32_t slot_count = 1) {
        configure(slot_count);
    }

    void configure(uint32_t slot_count) {
        if (slot_count == 0)
            throw std::invalid_argument("KVMem executor slot count must be > 0");
        std::lock_guard<std::mutex> lock(mu_);
        for (const auto &slot : slots_) {
            if (slot.busy)
                throw std::runtime_error(
                    "cannot reconfigure KVMem executor slots while a lease is active");
        }
        slots_.clear();
        slots_.resize(slot_count);
        for (uint32_t i = 0; i < slot_count; ++i) slots_[i].index = i;
    }

    // Fail-fast admission: a caller either gets an executor lineage now or a
    // deterministic backpressure error. HTTP/request queues belong above this
    // layer so the scheduler never blocks while holding runtime resources.
    Lease acquire(const std::string &session_id, uint64_t now) {
        if (session_id.empty())
            throw std::invalid_argument("KVMem scheduler session id is empty");
        std::lock_guard<std::mutex> lock(mu_);

        // Affinity first: reuse the already-mounted executor lineage.
        for (auto &slot : slots_) {
            if (slot.session_id == session_id && !slot.dirty) {
                if (slot.busy) {
                    ++backpressure_rejections_;
                    throw std::runtime_error(
                        "KVMem executor slot for session is already busy");
                }
                slot.busy = true;
                slot.last_used_at = now;
                ++slot.generation;
                ++warm_hits_;
                return make_lease_locked(slot, session_id, {}, true, false, false);
            }
        }

        // Prefer an unused idle slot, then LRU-idle eviction.
        Slot *chosen = nullptr;
        for (auto &slot : slots_) {
            if (!slot.busy && slot.session_id.empty()) {
                chosen = &slot;
                break;
            }
        }
        if (!chosen) {
            for (auto &slot : slots_) {
                if (slot.busy) continue;
                if (!chosen || slot.last_used_at < chosen->last_used_at ||
                    (slot.last_used_at == chosen->last_used_at &&
                     slot.index < chosen->index)) {
                    chosen = &slot;
                }
            }
        }
        if (!chosen) {
            ++backpressure_rejections_;
            throw std::runtime_error("all KVMem executor slots are busy");
        }

        const std::string evicted = chosen->session_id;
        const bool requires_reset = chosen->dirty;
        chosen->session_id = session_id;
        chosen->busy = true;
        chosen->last_used_at = now;
        ++chosen->generation;
        ++chosen->mounts;
        ++chosen->cold_mounts;
        ++cold_mounts_;
        if (requires_reset) ++forced_cold_resets_;
        return make_lease_locked(*chosen, session_id, evicted, false, true,
                                 requires_reset);
    }

    void release(const Lease &lease, bool success, uint64_t now) {
        std::lock_guard<std::mutex> lock(mu_);
        Slot &slot = require_lease_locked(lease);
        slot.busy = false;
        slot.active_lease_id = 0;
        slot.last_used_at = now;
        if (!success) {
            // Executor state after an exception is not trusted as a warm cache.
            // Quarantine it as dirty; the next user gets an explicit cold-reset
            // lease and must reconstruct state before warm reuse is possible.
            slot.session_id.clear();
            slot.dirty = true;
            ++slot.fault_count;
            ++faulted_releases_;
        } else if (lease.requires_cold_reset) {
            slot.dirty = false;
            ++slot.reset_count;
        }
    }

    std::optional<uint32_t> evict_session(const std::string &session_id) {
        std::lock_guard<std::mutex> lock(mu_);
        for (auto &slot : slots_) {
            if (slot.session_id != session_id) continue;
            if (slot.busy)
                throw std::runtime_error(
                    "cannot evict KVMem session while its executor slot is busy");
            const uint32_t index = slot.index;
            slot.session_id.clear();
            return index;
        }
        return std::nullopt;
    }

    void clear_idle() {
        std::lock_guard<std::mutex> lock(mu_);
        for (auto &slot : slots_) {
            if (!slot.busy) slot.session_id.clear();
        }
    }

    Info info() const {
        std::lock_guard<std::mutex> lock(mu_);
        Info out;
        out.slot_count = static_cast<uint32_t>(slots_.size());
        out.lease_sequence = lease_sequence_;
        out.warm_hits = warm_hits_;
        out.cold_mounts = cold_mounts_;
        out.backpressure_rejections = backpressure_rejections_;
        out.faulted_releases = faulted_releases_;
        out.forced_cold_resets = forced_cold_resets_;
        out.slots = slots_;
        for (const auto &slot : slots_) {
            if (slot.busy) ++out.busy_slots;
            if (!slot.session_id.empty()) ++out.mounted_sessions;
        }
        return out;
    }

private:
    Lease make_lease_locked(Slot &slot, const std::string &session_id,
                            const std::string &evicted, bool warm, bool cold,
                            bool requires_reset) {
        Lease lease;
        lease.slot = slot.index;
        lease.lease_id = ++lease_sequence_;
        slot.active_lease_id = lease.lease_id;
        lease.session_id = session_id;
        lease.evicted_session_id = evicted;
        lease.warm_hit = warm;
        lease.cold_mount = cold;
        lease.requires_cold_reset = requires_reset;
        return lease;
    }

    Slot &require_lease_locked(const Lease &lease) {
        if (!lease || lease.slot >= slots_.size())
            throw std::invalid_argument("invalid KVMem executor lease");
        Slot &slot = slots_[lease.slot];
        if (!slot.busy || slot.session_id != lease.session_id ||
            slot.active_lease_id != lease.lease_id)
            throw std::runtime_error("stale KVMem executor lease");
        return slot;
    }

    mutable std::mutex mu_;
    std::vector<Slot> slots_;
    uint64_t lease_sequence_ = 0;
    uint64_t warm_hits_ = 0;
    uint64_t cold_mounts_ = 0;
    uint64_t backpressure_rejections_ = 0;
    uint64_t faulted_releases_ = 0;
    uint64_t forced_cold_resets_ = 0;
};

} // namespace qw3::detail
