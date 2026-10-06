#include "kvmem_executor_slot_runtime.hpp"

#include <cassert>
#include <iostream>
#include <memory>

struct FakeExecutor { int marker = 7; };

int main() {
    using Runtime = qw3::detail::KvMemExecutorSlotRuntime<FakeExecutor>;
    Runtime rt(3);
    rt = std::make_unique<FakeExecutor>();
    assert(rt && rt->marker == 7);

    auto i0 = rt.info();
    assert(i0.slot == 3);
    assert(i0.runtime_id == "qw3-executor-slot-3");
    assert(i0.installed && !i0.busy && !i0.dirty);

    const auto lease_a1 = rt.bind("session-a", false, false, 10);
    assert(rt.info().busy);
    assert(rt.info().active_lease_id == lease_a1);
    rt.release(lease_a1, true, 11);
    assert(rt.info().mounted_session_id == "session-a");

    const auto lease_a2 = rt.bind("session-a", true, false, 12);
    rt.release(lease_a2, false, 13);
    auto failed = rt.info();
    assert(failed.dirty && failed.mounted_session_id.empty());
    assert(failed.faulted_releases == 1);

    const auto lease_b1 = rt.bind("session-b", false, true, 14);
    rt.mark_cold_reset(lease_b1, 15);
    rt.release(lease_b1, true, 16);
    auto recovered = rt.info();
    assert(!recovered.dirty);
    assert(recovered.mounted_session_id == "session-b");
    assert(recovered.cold_resets == 1);
    assert(recovered.leases == 3);

    // A stale physical lease cannot release a newer owner, even for the same
    // mounted session.
    const auto lease_b2 = rt.bind("session-b", true, false, 17, lease_b1);
    assert(lease_b2 > lease_b1);
    assert(rt.info().linked_scheduler_lease_id == lease_b1);
    bool stale_physical_release = false;
    try { rt.release(lease_b1, true, 17); }
    catch (...) { stale_physical_release = true; }
    assert(stale_physical_release);
    assert(rt.info().busy && rt.info().active_lease_id == lease_b2);
    rt.release(lease_b2, true, 17);

    bool bad_warm = false;
    try { (void)rt.bind("session-c", true, false, 17); }
    catch (...) { bad_warm = true; }
    assert(bad_warm);

    rt.invalidate_affinity(18);
    assert(rt.info().mounted_session_id.empty());

    std::cout << "kvmem_executor_slot_runtime_test: PASS\n";
    return 0;
}
