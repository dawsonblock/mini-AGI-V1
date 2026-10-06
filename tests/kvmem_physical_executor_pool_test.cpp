#include "kvmem_physical_executor_pool.hpp"

#include <cassert>
#include <iostream>
#include <memory>

struct FakeExecutor {
    explicit FakeExecutor(int value) : marker(value) {}
    int marker = 0;
};

int main() {
    using Pool = qw3::detail::KvMemPhysicalExecutorPool<FakeExecutor>;
    Pool pool(2);
    pool.install_all([](uint32_t slot) {
        return std::make_unique<FakeExecutor>(100 + static_cast<int>(slot));
    });
    assert(pool.executor(0) != pool.executor(1));
    assert(pool.executor(0)->marker == 100);
    assert(pool.executor(1)->marker == 101);

    auto &a = pool.runtime(0);
    auto &b = pool.runtime(1);
    const auto a_lease = a.bind("a", false, false, 1, 101);
    const auto b_lease = b.bind("b", false, false, 2, 202);
    assert(a_lease == 1 && b_lease == 1);
    assert(a.info().linked_scheduler_lease_id == 101);
    assert(b.info().linked_scheduler_lease_id == 202);
    auto both = pool.info();
    assert(both.configured_slots == 2);
    assert(both.installed_slots == 2);
    assert(both.busy_slots == 2);

    a.release(a_lease, true, 3);
    b.release(b_lease, false, 4);
    auto after = pool.info();
    assert(after.busy_slots == 0);
    assert(after.dirty_slots == 1);
    assert(after.slots[1].faulted_releases == 1);

    const auto c_lease = b.bind("c", false, true, 5, 203);
    assert(c_lease > b_lease);
    assert(b.info().linked_scheduler_lease_id == 203);
    b.mark_cold_reset(c_lease, 6);
    b.release(c_lease, true, 7);
    assert(pool.info().dirty_slots == 0);

    std::cout << "kvmem_physical_executor_pool_test: PASS\n";
    return 0;
}
