#pragma once

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace qw3::detail {

// Fail-fast resource reservation for independently mutable executor lineages.
//
// This deliberately operates on configured/derived byte envelopes rather than
// allocator internals. It is a control-plane guard: real CUDA allocation still
// remains authoritative, but a request cannot be admitted when its declared
// executor footprint would oversubscribe the configured runtime envelope.
class KvMemResourceAdmission {
public:
    struct Capacity {
        uint64_t vram_bytes = 0; // 0 => unbounded/not enforced
        uint64_t host_bytes = 0;
        uint64_t nvme_bytes = 0;
        uint32_t max_inflight = 1;
    };

    struct Request {
        uint64_t vram_bytes = 0;
        uint64_t host_bytes = 0;
        uint64_t nvme_bytes = 0;
        uint32_t slot = 0;
        std::string session_id;
    };

    struct Lease {
        uint64_t id = 0;
        Request request;
        explicit operator bool() const { return id != 0; }
    };

    struct Info {
        Capacity capacity;
        uint64_t used_vram_bytes = 0;
        uint64_t used_host_bytes = 0;
        uint64_t used_nvme_bytes = 0;
        uint64_t high_water_vram_bytes = 0;
        uint64_t high_water_host_bytes = 0;
        uint64_t high_water_nvme_bytes = 0;
        uint32_t inflight = 0;
        uint32_t high_water_inflight = 0;
        uint64_t admissions = 0;
        uint64_t rejections = 0;
    };

    KvMemResourceAdmission() { configure(Capacity{}); }

    explicit KvMemResourceAdmission(Capacity capacity) {
        configure(capacity);
    }

    void configure(Capacity capacity) {
        if (capacity.max_inflight == 0)
            throw std::invalid_argument("KVMem resource max_inflight must be > 0");
        std::lock_guard<std::mutex> lock(mu_);
        if (!leases_.empty())
            throw std::runtime_error(
                "cannot reconfigure KVMem resource admission with active leases");
        capacity_ = capacity;
        used_vram_bytes_ = used_host_bytes_ = used_nvme_bytes_ = 0;
        inflight_ = 0;
    }

    Lease acquire(const Request &request) {
        if (request.session_id.empty())
            throw std::invalid_argument("KVMem resource request session id is empty");
        std::lock_guard<std::mutex> lock(mu_);
        if (inflight_ >= capacity_.max_inflight ||
            exceeds(capacity_.vram_bytes, used_vram_bytes_, request.vram_bytes) ||
            exceeds(capacity_.host_bytes, used_host_bytes_, request.host_bytes) ||
            exceeds(capacity_.nvme_bytes, used_nvme_bytes_, request.nvme_bytes)) {
            ++rejections_;
            throw std::runtime_error("KVMem executor resource admission rejected request");
        }

        Lease lease;
        lease.id = ++lease_sequence_;
        lease.request = request;
        leases_.emplace(lease.id, lease.request);
        used_vram_bytes_ += request.vram_bytes;
        used_host_bytes_ += request.host_bytes;
        used_nvme_bytes_ += request.nvme_bytes;
        ++inflight_;
        ++admissions_;
        high_water_vram_bytes_ = std::max(high_water_vram_bytes_, used_vram_bytes_);
        high_water_host_bytes_ = std::max(high_water_host_bytes_, used_host_bytes_);
        high_water_nvme_bytes_ = std::max(high_water_nvme_bytes_, used_nvme_bytes_);
        high_water_inflight_ = std::max(high_water_inflight_, inflight_);
        return lease;
    }

    void release(const Lease &lease) {
        if (!lease)
            throw std::invalid_argument("invalid KVMem resource lease");
        std::lock_guard<std::mutex> lock(mu_);
        auto it = leases_.find(lease.id);
        if (it == leases_.end())
            throw std::runtime_error("stale KVMem resource lease");
        const Request &r = it->second;
        used_vram_bytes_ -= std::min(used_vram_bytes_, r.vram_bytes);
        used_host_bytes_ -= std::min(used_host_bytes_, r.host_bytes);
        used_nvme_bytes_ -= std::min(used_nvme_bytes_, r.nvme_bytes);
        if (inflight_ > 0) --inflight_;
        leases_.erase(it);
    }

    Info info() const {
        std::lock_guard<std::mutex> lock(mu_);
        Info out;
        out.capacity = capacity_;
        out.used_vram_bytes = used_vram_bytes_;
        out.used_host_bytes = used_host_bytes_;
        out.used_nvme_bytes = used_nvme_bytes_;
        out.high_water_vram_bytes = high_water_vram_bytes_;
        out.high_water_host_bytes = high_water_host_bytes_;
        out.high_water_nvme_bytes = high_water_nvme_bytes_;
        out.inflight = inflight_;
        out.high_water_inflight = high_water_inflight_;
        out.admissions = admissions_;
        out.rejections = rejections_;
        return out;
    }

private:
    static bool exceeds(uint64_t cap, uint64_t used, uint64_t add) {
        if (cap == 0) return false;
        return used > cap || add > cap - used;
    }

    mutable std::mutex mu_;
    Capacity capacity_;
    std::unordered_map<uint64_t, Request> leases_;
    uint64_t lease_sequence_ = 0;
    uint64_t used_vram_bytes_ = 0;
    uint64_t used_host_bytes_ = 0;
    uint64_t used_nvme_bytes_ = 0;
    uint64_t high_water_vram_bytes_ = 0;
    uint64_t high_water_host_bytes_ = 0;
    uint64_t high_water_nvme_bytes_ = 0;
    uint32_t inflight_ = 0;
    uint32_t high_water_inflight_ = 0;
    uint64_t admissions_ = 0;
    uint64_t rejections_ = 0;
};

} // namespace qw3::detail
