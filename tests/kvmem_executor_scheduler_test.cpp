#include "kvmem_executor_scheduler.hpp"

#include <cassert>
#include <iostream>

int main() {
    using qw3::detail::KvMemExecutorScheduler;
    KvMemExecutorScheduler sched(2);

    auto a1 = sched.acquire("a", 10);
    assert(a1.cold_mount && !a1.warm_hit && a1.slot == 0);
    sched.release(a1, true, 11);

    auto b1 = sched.acquire("b", 12);
    assert(b1.cold_mount && b1.slot == 1);
    sched.release(b1, true, 13);

    auto a2 = sched.acquire("a", 14);
    assert(a2.warm_hit && !a2.cold_mount && a2.slot == 0);
    sched.release(a2, true, 15);

    // Exact lease fencing: an older lease for the same session must not be able
    // to release a newer owner.
    auto a_stale_source = sched.acquire("a", 15);
    sched.release(a_stale_source, true, 15);
    auto a_new_owner = sched.acquire("a", 15);
    bool stale_release_rejected = false;
    try { sched.release(a_stale_source, true, 15); }
    catch (...) { stale_release_rejected = true; }
    assert(stale_release_rejected);
    assert(sched.info().busy_slots == 1);
    assert(sched.info().slots[a_new_owner.slot].active_lease_id == a_new_owner.lease_id);
    sched.release(a_new_owner, true, 15);

    // Slot 1 (b) is LRU and should be evicted for c.
    auto c1 = sched.acquire("c", 16);
    assert(c1.cold_mount && c1.slot == 1);
    assert(c1.evicted_session_id == "b");
    sched.release(c1, true, 17);

    // Fill both slots with active leases and prove fail-fast backpressure.
    auto a3 = sched.acquire("a", 18);
    auto c2 = sched.acquire("c", 18);
    bool rejected = false;
    try { (void)sched.acquire("d", 18); }
    catch (...) { rejected = true; }
    assert(rejected);
    sched.release(a3, true, 19);
    sched.release(c2, false, 19); // failed lineage is quarantined dirty

    // The dirty slot is reusable only through an explicit cold-reset lease.
    auto d1 = sched.acquire("d", 20);
    assert(d1.cold_mount && d1.requires_cold_reset);
    sched.release(d1, true, 21);

    auto info = sched.info();
    assert(info.slot_count == 2);
    assert(info.busy_slots == 0);
    assert(info.warm_hits >= 3);
    assert(info.cold_mounts >= 4);
    assert(info.backpressure_rejections == 1);
    assert(info.faulted_releases == 1);
    assert(info.forced_cold_resets == 1);
    assert(!info.slots[1].dirty);
    assert(info.slots[1].reset_count == 1);

    auto evicted = sched.evict_session("a");
    assert(evicted && *evicted == 0);
    auto evicted_d = sched.evict_session("d");
    assert(evicted_d && *evicted_d == 1);
    assert(sched.info().mounted_sessions == 0);

    std::cout << "kvmem_executor_scheduler_test: PASS\n";
    return 0;
}
