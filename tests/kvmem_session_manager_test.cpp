#include "kvmem_session_manager.hpp"

#include <cassert>
#include <iostream>

int main() {
    using qw3::detail::KvMemSessionManager;
    KvMemSessionManager mgr(2, 8, 256);

    auto a = mgr.start("a", "workspace-a", 10);
    a.tokens = {1,2,3};
    mgr.commit(a, 11);
    assert(!mgr.info("a").hot);
    mgr.mount("a", 0, false, 12);
    assert(mgr.info("a").hot);
    assert(mgr.info("a").executor_slot == 0);
    assert(mgr.info("a").token_count == 3);
    assert(mgr.info("a").host_bytes > 0);
    assert(mgr.total_host_bytes() == mgr.info("a").host_bytes);

    auto b = mgr.start("b", "workspace-b", 13);
    b.tokens = {4,5};
    mgr.commit(b, 14);
    // The registry is N-slot capable: a and b can both be mounted on distinct
    // executor lineages even though the current native backend certifies one.
    mgr.mount("b", 1, false, 15);
    assert(mgr.info("a").hot);
    assert(mgr.info("b").hot);
    assert(mgr.info("b").executor_slot == 1);

    auto old_a = mgr.require("a", "workspace-a", 16);
    mgr.unmount("a", 17);
    mgr.mount("a", 0, true, 18);
    assert(mgr.info("a").cold_rehydrates == 1);

    bool mismatch = false;
    try { (void)mgr.require("a", "wrong-workspace", 19); }
    catch (...) { mismatch = true; }
    assert(mismatch);

    bool omitted_workspace = false;
    try { (void)mgr.require("a", "", 19); }
    catch (...) { omitted_workspace = true; }
    assert(omitted_workspace);

    // Backend maintenance paths (snapshot/save) need a trusted lookup that does
    // not pretend an omitted caller workspace is authorized.
    const auto internal_a = mgr.require_internal("a", 19);
    assert(internal_a.workspace_id == "workspace-a");
    assert(internal_a.tokens.size() == 3);

    bool reset_mismatch = false;
    try { (void)mgr.start("a", "wrong-workspace", 20); }
    catch (...) { reset_mismatch = true; }
    assert(reset_mismatch);

    old_a.tokens.push_back(6);
    mgr.commit(old_a, 21);
    assert(mgr.info("a").token_count == 4);
    assert(mgr.info("a").executor_slot == 0); // stale Record cannot steal mount state

    bool max_sessions = false;
    try { (void)mgr.start("c", "workspace-c", 22); }
    catch (...) { max_sessions = true; }
    assert(max_sessions);

    const bool erased_b = mgr.erase("b");
    assert(erased_b);
    auto c = mgr.start("c", "workspace-c", 23);
    c.tokens = {7,8,9,10,11};
    bool token_limit = false;
    try { mgr.commit(c, 24); }
    catch (...) { token_limit = true; }
    assert(token_limit);

    KvMemSessionManager byte_mgr(2, 100, 32);
    auto d = byte_mgr.start("d", "w", 25);
    d.tokens = {1,2,3,4,5,6,7,8};
    bool byte_limit = false;
    try { byte_mgr.commit(d, 26); }
    catch (...) { byte_limit = true; }
    assert(byte_limit);

    byte_mgr.unmount_all(27);
    assert(!byte_mgr.info("d").hot);

    std::cout << "kvmem_session_manager_test: PASS\n";
    return 0;
}
