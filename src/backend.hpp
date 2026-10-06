#pragma once

#include "qw3/qw3.hpp"

#include <memory>
#include <stdexcept>
#include <string>

namespace qw3 {

class Backend {
public:
    virtual ~Backend() = default;
    virtual void load(const EngineOptions &options) = 0;
    virtual std::string generate(const std::string &prompt,
                                 const GenerationOptions &options,
                                 const CancellableTokenCallback &on_text) = 0;
    virtual std::string generate_session(const std::string &prompt_fragment,
                                         const GenerationOptions &options,
                                         const TokenCallback &on_text,
                                         bool reset) {
        if (reset) {
            return generate(
                prompt_fragment, options,
                on_text ? CancellableTokenCallback(
                              [on_text](const std::string &piece) {
                                  on_text(piece);
                                  return true;
                              })
                        : CancellableTokenCallback{});
        }
        throw std::runtime_error(
            "persistent session append is unsupported by this backend");
    }
    virtual KvMemLocalCacheInfo kvmem_local_cache_info(
            const std::string &id) {
        (void)id;
        return {};
    }
    virtual bool erase_kvmem_local_cache(const std::string &id) {
        (void)id;
        return false;
    }
    virtual KvMemSessionInfo kvmem_session_info(const std::string &id) {
        (void)id;
        return {};
    }
    virtual std::vector<KvMemSessionInfo> kvmem_session_infos() {
        return {};
    }
    virtual KvMemExecutorSchedulerInfo kvmem_executor_scheduler_info() {
        return {};
    }
    virtual KvMemResourceAdmissionInfo kvmem_resource_admission_info() {
        return {};
    }
    virtual KvMemPhysicalExecutorPoolInfo kvmem_physical_executor_pool_info() {
        return {};
    }
    virtual bool erase_kvmem_session(const std::string &id) {
        (void)id;
        return false;
    }
    virtual KvMemSessionSnapshotInfo snapshot_kvmem_session(
            const std::string &id) {
        (void)id;
        throw std::runtime_error("KVMem session snapshots are unsupported by this backend");
    }
    virtual KvMemSessionSnapshotInfo restore_kvmem_session_snapshot(
            const std::string &id) {
        (void)id;
        throw std::runtime_error("KVMem session snapshots are unsupported by this backend");
    }
    virtual KvMemSessionSnapshotInfo kvmem_session_snapshot_info(
            const std::string &id) {
        (void)id;
        return {};
    }
    virtual VisionEncoding encode_vision(
            const std::vector<VisionImage> &images) {
        (void)images;
        throw std::runtime_error("native CUDA vision is not configured");
    }
};

std::unique_ptr<Backend> make_qwen_native_backend();

} // namespace qw3
