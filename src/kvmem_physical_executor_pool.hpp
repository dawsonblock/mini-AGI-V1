#pragma once

#include "kvmem_executor_slot_runtime.hpp"

#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace qw3::detail {

// Physical pool of independently owned mutable executor lineages.
//
// The pool is deliberately construction-policy agnostic: the backend supplies a
// factory for each slot, so every installed runtime owns a distinct ExecutorT.
// Shared immutable model weights/device services may be captured by the factory,
// but live executor/KV/recurrent state is never shared through this class.
template <class ExecutorT>
class KvMemPhysicalExecutorPool {
public:
    using Runtime = KvMemExecutorSlotRuntime<ExecutorT>;

    struct Info {
        uint32_t configured_slots = 0;
        uint32_t installed_slots = 0;
        uint32_t busy_slots = 0;
        uint32_t dirty_slots = 0;
        std::vector<typename Runtime::Info> slots;
    };

    explicit KvMemPhysicalExecutorPool(uint32_t slots = 1) { configure(slots); }

    void configure(uint32_t slots) {
        if (slots == 0) throw std::invalid_argument("physical executor pool slots must be > 0");
        std::lock_guard<std::mutex> lock(mu_);
        for (const auto &rt : slots_) {
            if (rt && rt->info().busy)
                throw std::runtime_error("cannot reconfigure physical executor pool while busy");
        }
        slots_.clear();
        slots_.reserve(slots);
        for (uint32_t i = 0; i < slots; ++i)
            slots_.push_back(std::make_unique<Runtime>(i));
    }

    template <class Factory>
    void install_all(Factory &&factory) {
        std::lock_guard<std::mutex> lock(mu_);
        for (uint32_t i = 0; i < slots_.size(); ++i) {
            if (slots_[i]->info().installed) continue;
            auto executor = factory(i);
            if (!executor) throw std::runtime_error("physical executor factory returned null");
            slots_[i]->install(std::move(executor));
        }
    }

    void install(uint32_t slot, std::unique_ptr<ExecutorT> executor) {
        runtime(slot).install(std::move(executor));
    }

    Runtime &runtime(uint32_t slot) {
        std::lock_guard<std::mutex> lock(mu_);
        if (slot >= slots_.size()) throw std::out_of_range("physical executor slot out of range");
        return *slots_[slot];
    }
    const Runtime &runtime(uint32_t slot) const {
        std::lock_guard<std::mutex> lock(mu_);
        if (slot >= slots_.size()) throw std::out_of_range("physical executor slot out of range");
        return *slots_[slot];
    }

    ExecutorT *executor(uint32_t slot) { return runtime(slot).get(); }
    const ExecutorT *executor(uint32_t slot) const { return runtime(slot).get(); }

    Info info() const {
        std::lock_guard<std::mutex> lock(mu_);
        Info out;
        out.configured_slots = static_cast<uint32_t>(slots_.size());
        out.slots.reserve(slots_.size());
        for (const auto &rt : slots_) {
            auto item = rt->info();
            if (item.installed) ++out.installed_slots;
            if (item.busy) ++out.busy_slots;
            if (item.dirty) ++out.dirty_slots;
            out.slots.push_back(std::move(item));
        }
        return out;
    }

private:
    mutable std::mutex mu_;
    std::vector<std::unique_ptr<Runtime>> slots_;
};

} // namespace qw3::detail
