#pragma once

#include "qw3/qw3.hpp"
#include "server.hpp"

#include <stdexcept>
#include <string>

namespace qw3::detail {

inline bool valid_kvmem_profile(const std::string &profile) {
    return profile == "custom" || profile == "agent-safe" ||
           profile == "agent-fast" || profile == "throughput" ||
           profile == "research";
}

inline bool valid_kvmem_state_coherence(const std::string &mode) {
    return mode == "attention-kv" || mode == "query-replay" ||
           mode == "selected-replay";
}

// Profiles are presets, not hidden policy. The CLI pre-applies these defaults
// before normal option parsing, so later explicit flags may override them. The
// final validator below then rejects combinations that violate the declared
// profile contract.
inline void apply_kvmem_profile_defaults(const std::string &profile,
                                         EngineOptions &engine,
                                         ServerConfig &server) {
    if (!valid_kvmem_profile(profile)) {
        throw std::runtime_error(
            "--kvmem-profile must be custom|agent-safe|agent-fast|throughput|research");
    }
    engine.kvmem_profile = profile;
    if (profile == "custom" || profile == "research") return;

    engine.kvmem_enabled = true;
    engine.kvmem_method = "retrieval";
    engine.kvmem_retrieval_method = "mean-k";
    engine.kvmem_query_conditioned = true;
    engine.kvmem_recompute_query = true;
    engine.kvmem_immutable_source_k = true;

    if (profile == "agent-safe") {
        engine.kvmem_strict_retrieval = true;
        engine.kvmem_update_mode = "step";
        engine.kvmem_semantic_expansion = "message";
        engine.kvmem_state_coherence = "selected-replay";
        server.kvmem_query_replay = true;
        server.continuous_batching = false;
        return;
    }
    if (profile == "agent-fast") {
        engine.kvmem_strict_retrieval = true;
        engine.kvmem_update_mode = "step";
        engine.kvmem_semantic_expansion = "message";
        engine.kvmem_state_coherence = "query-replay";
        server.kvmem_query_replay = true;
        server.continuous_batching = false;
        return;
    }
    if (profile == "throughput") {
        engine.kvmem_state_coherence = "attention-kv";
        server.kvmem_query_replay = false;
        server.continuous_batching = true;
        return;
    }
}

inline void validate_kvmem_runtime_contract(const EngineOptions &engine,
                                            const ServerConfig &server) {
    if (!valid_kvmem_profile(engine.kvmem_profile)) {
        throw std::runtime_error("invalid KVMem runtime profile: " +
                                 engine.kvmem_profile);
    }
    if (!valid_kvmem_state_coherence(engine.kvmem_state_coherence)) {
        throw std::runtime_error("invalid KVMem state coherence mode: " +
                                 engine.kvmem_state_coherence);
    }
    if (engine.kvmem_state_coherence != "attention-kv") {
        if (!engine.kvmem_enabled || !engine.kvmem_query_conditioned) {
            throw std::runtime_error(
                "KVMem state coherence query-replay/selected-replay requires "
                "--kvmem and --kvmem-query-conditioned");
        }
        if (!server.kvmem_query_replay) {
            throw std::runtime_error(
                "KVMem state coherence query-replay/selected-replay requires "
                "--kvmem-query-replay");
        }
        if (server.continuous_batching) {
            throw std::runtime_error(
                "KVMem state coherence query-replay/selected-replay is "
                "single-request only and cannot use --continuous-batching");
        }
    }
    if (engine.kvmem_state_coherence == "selected-replay" &&
        !engine.kvmem_strict_retrieval) {
        throw std::runtime_error(
            "selected-replay requires --kvmem-strict-retrieval so a failed "
            "retrieval scorer cannot silently change the rebuilt context");
    }

    if (engine.kvmem_profile == "agent-safe") {
        if (!engine.kvmem_enabled || !engine.kvmem_query_conditioned ||
            !engine.kvmem_strict_retrieval ||
            !engine.kvmem_immutable_source_k ||
            engine.kvmem_method != "retrieval" ||
            engine.kvmem_retrieval_method != "mean-k" ||
            engine.kvmem_update_mode != "step" ||
            engine.kvmem_semantic_expansion != "message" ||
            engine.kvmem_state_coherence != "selected-replay" ||
            !server.kvmem_query_replay || server.continuous_batching) {
            throw std::runtime_error(
                "agent-safe profile contract was overridden with an "
                "uncertified combination; use --kvmem-profile custom or "
                "research for experiments");
        }
    } else if (engine.kvmem_profile == "agent-fast") {
        if (!engine.kvmem_enabled || !engine.kvmem_query_conditioned ||
            !engine.kvmem_strict_retrieval ||
            !engine.kvmem_immutable_source_k ||
            engine.kvmem_method != "retrieval" ||
            engine.kvmem_retrieval_method != "mean-k" ||
            engine.kvmem_update_mode != "step" ||
            engine.kvmem_semantic_expansion != "message" ||
            engine.kvmem_state_coherence != "query-replay" ||
            !server.kvmem_query_replay || server.continuous_batching) {
            throw std::runtime_error(
                "agent-fast profile contract was overridden with an "
                "uncertified combination; use --kvmem-profile custom or "
                "research for experiments");
        }
    } else if (engine.kvmem_profile == "throughput") {
        if (!engine.kvmem_enabled || !server.continuous_batching ||
            engine.kvmem_state_coherence != "attention-kv") {
            throw std::runtime_error(
                "throughput profile requires KVMem continuous batching with "
                "attention-kv coherence; use custom/research to override");
        }
    }
}

inline bool kvmem_hybrid_state_exact_for_request(const EngineOptions &engine) {
    // selected-replay densely rebuilds both normal-attention KV and recurrent
    // state from the selected source tokens for standalone one-shot requests.
    // Other sparse modes retain a hybrid recurrent state that is not generally
    // equivalent to arbitrary selected history (KVMI-012).
    return !engine.kvmem_enabled ||
           engine.kvmem_state_coherence == "selected-replay";
}

} // namespace qw3::detail
