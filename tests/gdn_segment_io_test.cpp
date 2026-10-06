#include "qw3/gdn_segment_io.hpp"

#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

int main() {
    qw3::GdnPreparedCapture c;
    c.layer_index = 7;
    c.position_start = 1234;
    c.T = 11;
    c.num_k_heads = 2;
    c.num_v_heads = 4;
    c.head_dim = 16;
    c.model_id = "unit-test-qwen";
    c.q.resize(static_cast<size_t>(c.T) * c.num_k_heads * c.head_dim);
    c.k.resize(c.q.size());
    c.v.resize(static_cast<size_t>(c.T) * c.num_v_heads * c.head_dim);
    c.decay.resize(static_cast<size_t>(c.T) * c.num_v_heads);
    c.beta.resize(c.decay.size());
    c.conv_kernel_size = 4;
    c.conv_dim = 2u * c.num_k_heads * c.head_dim + c.num_v_heads * c.head_dim;
    c.preconv.resize(static_cast<size_t>(c.T) * c.conv_dim);
    for (size_t i = 0; i < c.q.size(); ++i) {
        c.q[i] = 0.05f * std::sin(float(i + 1));
        c.k[i] = 0.04f * std::cos(float(i + 3));
    }
    for (size_t i = 0; i < c.v.size(); ++i) c.v[i] = 0.03f * std::sin(float(i + 7) * 0.31f);
    for (size_t i = 0; i < c.decay.size(); ++i) {
        c.decay[i] = 0.91f + 0.0005f * float(i % 20);
        c.beta[i] = 0.1f + 0.01f * float(i % 5);
    }
    for (size_t i = 0; i < c.preconv.size(); ++i) {
        c.preconv[i] = 0.017f * std::sin(float(i + 11) * 0.13f);
    }
    c.validate();

    const auto tmp = std::filesystem::temp_directory_path();
    const auto cap = (tmp / "qw3_gdn_segment_io_test.qgdn").string();
    const auto sum = (tmp / "qw3_gdn_segment_io_test.qgds").string();
    qw3::write_gdn_capture(cap, c);
    const auto r = qw3::read_gdn_capture(cap);
    assert(r.layer_index == c.layer_index && r.position_start == c.position_start);
    assert(r.k == c.k && r.v == c.v && r.decay == c.decay && r.beta == c.beta);
    assert(r.version == 2 && r.conv_kernel_size == 4 && r.conv_dim == c.conv_dim);
    assert(r.preconv == c.preconv);

    const auto summaries = qw3::gdn_summarize_capture_reference(r, 4);
    assert(summaries.size() == 3);
    assert(summaries[0].position_start == 1234 && summaries[0].token_count == 4);
    assert(summaries[2].position_start == 1242 && summaries[2].token_count == 3);
    qw3::write_gdn_summary_archive(sum, summaries);
    const auto reread = qw3::read_gdn_summary_archive(sum);
    assert(reread.size() == summaries.size());
    assert(reread[0].version == 3);
    assert(reread[0].T == summaries[0].T && reread[0].Z == summaries[0].Z);
    assert(reread[0].conv_kernel_size == 4);
    assert(reread[0].conv_tail_rows == 3);
    assert(reread[0].conv_tail == summaries[0].conv_tail);

    // Integrity must fail closed on payload corruption.
    {
        std::fstream f(cap, std::ios::in | std::ios::out | std::ios::binary);
        f.seekg(32);
        char x = 0;
        f.read(&x, 1);
        f.seekp(32);
        x ^= 0x5a;
        f.write(&x, 1);
    }
    bool rejected = false;
    try { (void)qw3::read_gdn_capture(cap); }
    catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
    std::filesystem::remove(cap);
    std::filesystem::remove(sum);
    return 0;
}
