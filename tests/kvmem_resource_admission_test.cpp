#include "kvmem_resource_admission.hpp"

#include <cassert>
#include <iostream>

int main() {
    using qw3::detail::KvMemResourceAdmission;
    KvMemResourceAdmission::Capacity cap;
    cap.vram_bytes = 100;
    cap.host_bytes = 200;
    cap.nvme_bytes = 300;
    cap.max_inflight = 2;
    KvMemResourceAdmission admission(cap);

    KvMemResourceAdmission::Request a;
    a.vram_bytes = 40; a.host_bytes = 50; a.nvme_bytes = 60;
    a.slot = 0; a.session_id = "a";
    auto la = admission.acquire(a);

    KvMemResourceAdmission::Request b;
    b.vram_bytes = 50; b.host_bytes = 100; b.nvme_bytes = 200;
    b.slot = 1; b.session_id = "b";
    auto lb = admission.acquire(b);

    bool inflight_rejected = false;
    try { (void)admission.acquire(a); }
    catch (...) { inflight_rejected = true; }
    assert(inflight_rejected);

    admission.release(la);
    KvMemResourceAdmission::Request c;
    c.vram_bytes = 60; c.host_bytes = 101; c.nvme_bytes = 101;
    c.slot = 0; c.session_id = "c";
    bool bytes_rejected = false;
    try { (void)admission.acquire(c); }
    catch (...) { bytes_rejected = true; }
    assert(bytes_rejected);

    admission.release(lb);
    auto lc = admission.acquire(c);
    admission.release(lc);

    auto info = admission.info();
    assert(info.inflight == 0);
    assert(info.admissions == 3);
    assert(info.rejections == 2);
    assert(info.high_water_inflight == 2);
    assert(info.high_water_vram_bytes == 90);

    std::cout << "kvmem_resource_admission_test: PASS\n";
    return 0;
}
