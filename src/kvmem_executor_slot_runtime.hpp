#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

namespace qw3::detail {

// Physical ownership boundary for one independently mutable executor lineage.
//
// A logical KVMem session is portable host state. An ExecutorSlotRuntime is not:
// it owns the concrete executor object and therefore all live KV/recurrent/tier
// state reachable from that executor. This wrapper deliberately contains no
// CUDA-specific code; it provides lifecycle fencing that can be unit tested
// with a fake executor and then used by the native CUDA backend.
template <class ExecutorT>
class KvMemExecutorSlotRuntime {
public:
    struct Info {
        uint32_t slot = 0;
        std::string runtime_id;
        bool installed = false;
        bool busy = false;
        bool dirty = false;
        std::string mounted_session_id;
        uint64_t runtime_generation = 0;
        uint64_t leases = 0;
        uint64_t successful_releases = 0;
        uint64_t faulted_releases = 0;
        uint64_t cold_resets = 0;
        uint64_t last_used_at = 0;
        uint64_t active_lease_id = 0;
        uint64_t lease_sequence = 0;
        uint64_t linked_scheduler_lease_id = 0;
    };

    explicit KvMemExecutorSlotRuntime(uint32_t slot = 0)
        : slot_(slot), runtime_id_("qw3-executor-slot-" + std::to_string(slot)) {}

    KvMemExecutorSlotRuntime(const KvMemExecutorSlotRuntime &) = delete;
    KvMemExecutorSlotRuntime &operator=(const KvMemExecutorSlotRuntime &) = delete;

    KvMemExecutorSlotRuntime &operator=(std::unique_ptr<ExecutorT> executor) {
        install(std::move(executor));
        return *this;
    }

    void install(std::unique_ptr<ExecutorT> executor) {
        if (!executor) throw std::invalid_argument("executor runtime cannot install null executor");
        std::lock_guard<std::mutex> lock(mu_);
        if (executor_) throw std::runtime_error("executor runtime already installed");
        executor_ = std::move(executor);
        ++runtime_generation_;
    }

    ExecutorT *get() noexcept { return executor_.get(); }
    const ExecutorT *get() const noexcept { return executor_.get(); }
    ExecutorT *operator->() noexcept { return executor_.get(); }
    const ExecutorT *operator->() const noexcept { return executor_.get(); }
    ExecutorT &operator*() {
        if (!executor_) throw std::runtime_error("executor runtime is not installed");
        return *executor_;
    }
    const ExecutorT &operator*() const {
        if (!executor_) throw std::runtime_error("executor runtime is not installed");
        return *executor_;
    }
    explicit operator bool() const noexcept { return static_cast<bool>(executor_); }

    // Bind a scheduler lease to this physical runtime. Warm reuse is allowed
    // only when both scheduler affinity and physical runtime affinity agree.
    uint64_t bind(const std::string &session_id, bool warm_hit,
                  bool requires_cold_reset, uint64_t now,
                  uint64_t scheduler_lease_id = 0) {
        if (session_id.empty()) throw std::invalid_argument("executor runtime session id is empty");
        std::lock_guard<std::mutex> lock(mu_);
        if (!executor_) throw std::runtime_error("executor runtime is not installed");
        if (busy_) throw std::runtime_error("executor runtime is already busy");
        if (warm_hit && (dirty_ || mounted_session_id_ != session_id)) {
            throw std::runtime_error(
                "scheduler warm hit disagrees with physical executor affinity");
        }
        if (requires_cold_reset && !dirty_) {
            // Scheduler may conservatively require a reset after metadata loss.
            // Accept it; the physical runtime remains cold until mark_cold_reset().
        }
        mounted_session_id_ = session_id;
        busy_ = true;
        last_used_at_ = now;
        ++leases_;
        const uint64_t next_lease = ++lease_sequence_;
        active_lease_id_ = next_lease;
        linked_scheduler_lease_id_ = scheduler_lease_id;
        return next_lease;
    }

    void mark_cold_reset(uint64_t lease_id, uint64_t now) {
        std::lock_guard<std::mutex> lock(mu_);
        require_active_lease_locked(lease_id);
        dirty_ = false;
        last_used_at_ = now;
        ++cold_resets_;
        ++runtime_generation_;
    }

    void release(uint64_t lease_id, bool success, uint64_t now) {
        std::lock_guard<std::mutex> lock(mu_);
        require_active_lease_locked(lease_id);
        busy_ = false;
        active_lease_id_ = 0;
        linked_scheduler_lease_id_ = 0;
        last_used_at_ = now;
        if (success) {
            ++successful_releases_;
        } else {
            dirty_ = true;
            mounted_session_id_.clear();
            ++faulted_releases_;
            ++runtime_generation_;
        }
    }

    void invalidate_affinity(uint64_t now) {
        std::lock_guard<std::mutex> lock(mu_);
        if (busy_) throw std::runtime_error("cannot invalidate busy executor runtime");
        mounted_session_id_.clear();
        last_used_at_ = now;
    }

    Info info() const {
        std::lock_guard<std::mutex> lock(mu_);
        Info out;
        out.slot = slot_;
        out.runtime_id = runtime_id_;
        out.installed = static_cast<bool>(executor_);
        out.busy = busy_;
        out.dirty = dirty_;
        out.mounted_session_id = mounted_session_id_;
        out.runtime_generation = runtime_generation_;
        out.leases = leases_;
        out.successful_releases = successful_releases_;
        out.faulted_releases = faulted_releases_;
        out.cold_resets = cold_resets_;
        out.last_used_at = last_used_at_;
        out.active_lease_id = active_lease_id_;
        out.lease_sequence = lease_sequence_;
        out.linked_scheduler_lease_id = linked_scheduler_lease_id_;
        return out;
    }

private:
    void require_active_lease_locked(uint64_t lease_id) const {
        if (!executor_) throw std::runtime_error("executor runtime is not installed");
        if (!busy_ || lease_id == 0 || active_lease_id_ != lease_id)
            throw std::runtime_error("stale physical executor lease");
    }
    uint32_t slot_ = 0;
    std::string runtime_id_;
    std::unique_ptr<ExecutorT> executor_;
    mutable std::mutex mu_;
    bool busy_ = false;
    bool dirty_ = false;
    std::string mounted_session_id_;
    uint64_t runtime_generation_ = 0;
    uint64_t leases_ = 0;
    uint64_t successful_releases_ = 0;
    uint64_t faulted_releases_ = 0;
    uint64_t cold_resets_ = 0;
    uint64_t last_used_at_ = 0;
    uint64_t active_lease_id_ = 0;
    uint64_t lease_sequence_ = 0;
    uint64_t linked_scheduler_lease_id_ = 0;
};

} // namespace qw3::detail
