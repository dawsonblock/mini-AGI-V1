#pragma once

#include "qw3/qw3.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace qw3::detail {

// Detachable canonical host-side state for one logical KVMem trajectory.
//
// This is deliberately free of QwenExecutor snapshots, CUDA pointers, KV-page
// ownership, DeltaNet device state, MTP state and NVMe file descriptors. It is
// therefore safe to copy, persist (subject to embedding validation), move
// between schedulers and rebuild on another compatible executor. It remains a
// derived cache: the caller's durable WorkspaceLedger is authoritative.
struct KvMemSessionState {
    std::vector<uint32_t> tokens;
    uint64_t input_embedding_fingerprint = 0;
    std::vector<GenerationOptions::InputEmbeddingOverride>
        input_embedding_overrides;
    std::vector<std::array<uint32_t, 3>> input_mrope_positions;
    std::vector<GenerationOptions::KvMemPinnedTokenSpan>
        pinned_token_spans;

    bool empty() const { return tokens.empty(); }

    uint64_t estimated_host_bytes() const {
        uint64_t bytes = 0;
        bytes += static_cast<uint64_t>(tokens.size()) * sizeof(uint32_t);
        bytes += static_cast<uint64_t>(input_mrope_positions.size()) *
                 sizeof(std::array<uint32_t, 3>);
        bytes += static_cast<uint64_t>(pinned_token_spans.size()) *
                 sizeof(GenerationOptions::KvMemPinnedTokenSpan);
        for (const auto &override : input_embedding_overrides) {
            bytes += sizeof(override.token_id) + sizeof(override.source_token_id) +
                     sizeof(override.position) + sizeof(override.embedding_row);
            bytes += static_cast<uint64_t>(override.embedding.size()) * sizeof(float);
            if (override.storage) {
                // Admission accounting only. Device-backed rows are charged at
                // their logical FP32 payload size; snapshots reject such rows.
                bytes += static_cast<uint64_t>(override.storage->rows()) *
                         override.storage->dim() * sizeof(float);
            }
        }
        return bytes;
    }

    bool has_device_only_embedding_state() const {
        for (const auto &override : input_embedding_overrides) {
            if (override.storage) return true;
        }
        return false;
    }
};

} // namespace qw3::detail
