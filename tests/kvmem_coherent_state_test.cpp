#include "qw3/kvmem_coherent_state.hpp"

#include <cassert>
#include <cmath>
#include <stdexcept>
#include <vector>

using namespace qw3;

static KvMemCacheIdentity ident(const char* adapter = "a", const char* pos = "rope", const char* rec = "gdn-v1") {
    return {"model", adapter, "tok", "layout", pos, rec};
}

static KvMemAffineSummary affine2(double a, double b, double z0, double z1) {
    KvMemAffineSummary s;
    s.dim = 2;
    s.T = {a, 0.0, 0.0, b};
    s.Z = {z0, z1};
    return s;
}

int main() {
    {
        auto A = affine2(0.5, 0.75, 1.0, -1.0);
        auto B = affine2(0.8, 0.6, 0.25, 2.0);
        const std::vector<double> s0{2.0, 3.0};
        const auto direct = B.apply(A.apply(s0));
        const auto composed = A.then(B).apply(s0);
        assert(std::fabs(direct[0] - composed[0]) < 1e-12);
        assert(std::fabs(direct[1] - composed[1]) < 1e-12);
    }

    {
        const auto x = ident("adapter-a");
        const auto y = ident("adapter-b");
        assert(compare_cache_identity(x, x) == KvMemCacheCompatibility::ExactCompatible);
        assert(compare_cache_identity(x, y) == KvMemCacheCompatibility::ReplayRequired);
        auto p = x; p.position_scheme = "rope-v2";
        assert(compare_cache_identity(x, p) == KvMemCacheCompatibility::Invalid);
        auto r = x; r.recurrence_impl = "gdn-v2";
        assert(compare_cache_identity(x, r) == KvMemCacheCompatibility::Invalid);
    }

    {
        const auto id = ident();
        KvMemHybridBlockSummary A{1, id, {{0, affine2(.9,.8,1,2)}, {4, affine2(.7,.6,3,4)}}};
        KvMemHybridBlockSummary D{4, id, {{0, affine2(.5,.4,5,6)}, {4, affine2(.3,.2,7,8)}}};
        const auto out = compose_hybrid_blocks({A, D}, id);
        assert(out.selected_block_ids == std::vector<uint32_t>({1,4}));
        assert(out.layers.size() == 2);

        KvMemHybridBlockSummary bad{7, id, {{0, affine2(.5,.4,5,6)}}};
        bool threw = false;
        try { (void)compose_hybrid_blocks({A, bad}, id); }
        catch (const std::invalid_argument&) { threw = true; }
        assert(threw); // never silently drop layer 4
    }

    {
        KvMemAdaptiveRepairPolicy policy;
        KvMemCoherenceMetrics bad; bad.logit_kl = 0.5;
        KvMemCoherenceMetrics good; good.logit_kl = 0.001;
        auto d = policy.decide({bad, good}, {});
        assert(d.action == KvMemReconstructionAction::SeamReplay);
        assert(d.replay_tokens == 16);
        d = policy.decide({bad, bad, bad, bad, bad}, {bad, good});
        assert(d.action == KvMemReconstructionAction::SuffixReplay);
        assert(d.replay_tokens == 512);
        d = policy.decide({bad, bad, bad, bad, bad}, {bad, bad, bad, bad});
        assert(d.action == KvMemReconstructionAction::ExactSelectedReplay);
    }

    return 0;
}
