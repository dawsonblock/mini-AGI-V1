#include "kvmem_runtime_profile.hpp"

#include <cassert>
#include <stdexcept>

int main() {
    using namespace qw3;

    {
        EngineOptions e;
        ServerConfig s;
        detail::apply_kvmem_profile_defaults("agent-safe", e, s);
        assert(e.kvmem_enabled);
        assert(e.kvmem_strict_retrieval);
        assert(e.kvmem_query_conditioned);
        assert(e.kvmem_immutable_source_k);
        assert(e.kvmem_retrieval_method == "mean-k");
        assert(e.kvmem_update_mode == "step");
        assert(e.kvmem_semantic_expansion == "message");
        assert(e.kvmem_state_coherence == "selected-replay");
        assert(s.kvmem_query_replay);
        assert(!s.continuous_batching);
        detail::validate_kvmem_runtime_contract(e, s);
    }

    {
        EngineOptions e;
        ServerConfig s;
        detail::apply_kvmem_profile_defaults("agent-fast", e, s);
        assert(e.kvmem_state_coherence == "query-replay");
        detail::validate_kvmem_runtime_contract(e, s);
    }

    {
        EngineOptions e;
        ServerConfig s;
        detail::apply_kvmem_profile_defaults("throughput", e, s);
        assert(e.kvmem_state_coherence == "attention-kv");
        assert(s.continuous_batching);
        detail::validate_kvmem_runtime_contract(e, s);
    }

    {
        EngineOptions e;
        ServerConfig s;
        detail::apply_kvmem_profile_defaults("agent-safe", e, s);
        e.kvmem_strict_retrieval = false;
        bool threw = false;
        try {
            detail::validate_kvmem_runtime_contract(e, s);
        } catch (const std::runtime_error &) {
            threw = true;
        }
        assert(threw);
    }

    {
        EngineOptions e;
        ServerConfig s;
        e.kvmem_enabled = true;
        e.kvmem_query_conditioned = true;
        e.kvmem_strict_retrieval = true;
        e.kvmem_state_coherence = "selected-replay";
        s.kvmem_query_replay = true;
        s.continuous_batching = true;
        bool threw = false;
        try {
            detail::validate_kvmem_runtime_contract(e, s);
        } catch (const std::runtime_error &) {
            threw = true;
        }
        assert(threw);
    }

    return 0;
}
