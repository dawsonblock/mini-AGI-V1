#include "qw3/gdn_metal.hpp"
#include "qw3/gdn_segment_io.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<qw3::GdnBlockAffineSummary> summarize_metal(
    const qw3::GdnPreparedCapture& c, uint32_t block_tokens) {
    if (!qw3::gdn_metal_available()) {
        throw std::runtime_error("Metal summary requested but no Apple Metal device is available");
    }
    std::vector<qw3::GdnBlockAffineSummary> out;
    const uint32_t D = c.head_dim;
    for (uint32_t begin = 0; begin < c.T; begin += block_tokens) {
        const uint32_t n = std::min(block_tokens, c.T - begin);
        qw3::GdnMetalPreparedInputs in;
        in.T = n;
        in.num_k_heads = c.num_k_heads;
        in.num_v_heads = c.num_v_heads;
        in.head_dim = D;
        const size_t krow = static_cast<size_t>(c.num_k_heads) * D;
        const size_t vrow = static_cast<size_t>(c.num_v_heads) * D;
        const size_t grow = c.num_v_heads;
        in.k.assign(c.k.begin() + static_cast<size_t>(begin) * krow,
                    c.k.begin() + static_cast<size_t>(begin + n) * krow);
        in.v.assign(c.v.begin() + static_cast<size_t>(begin) * vrow,
                    c.v.begin() + static_cast<size_t>(begin + n) * vrow);
        in.decay.assign(c.decay.begin() + static_cast<size_t>(begin) * grow,
                        c.decay.begin() + static_cast<size_t>(begin + n) * grow);
        in.beta.assign(c.beta.begin() + static_cast<size_t>(begin) * grow,
                       c.beta.begin() + static_cast<size_t>(begin + n) * grow);
        auto m = qw3::gdn_metal_summarize(in);
        qw3::GdnBlockAffineSummary s;
        s.layer_index = c.layer_index;
        s.position_start = c.position_start + begin;
        s.token_count = n;
        s.num_v_heads = c.num_v_heads;
        s.head_dim = D;
        s.model_id = c.model_id;
        s.T = std::move(m.T);
        s.Z = std::move(m.Z);
        s.producer = "metal:" + m.device_name;
        s.validate();
        out.push_back(std::move(s));
    }
    return out;
}

double max_abs(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) throw std::runtime_error("summary comparison shape mismatch");
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        m = std::max(m, std::abs(static_cast<double>(a[i]) - b[i]));
    }
    return m;
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 3) {
            std::cerr << "Usage: qw3-gdn-summarize <capture.qgdn> <out.qgds> "
                         "[--block-tokens N] [--metal|--reference|--auto] [--verify]\n";
            return 2;
        }
        std::string capture_path = argv[1];
        std::string output_path = argv[2];
        uint32_t block_tokens = 16;
        enum class Mode { Auto, Metal, Reference } mode = Mode::Auto;
        bool verify = false;
        for (int i = 3; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--block-tokens") {
                if (++i >= argc) throw std::runtime_error("--block-tokens requires a value");
                const unsigned long v = std::stoul(argv[i]);
                if (v == 0 || v > (1u << 20)) throw std::runtime_error("invalid block token count");
                block_tokens = static_cast<uint32_t>(v);
            } else if (arg == "--metal") mode = Mode::Metal;
            else if (arg == "--reference") mode = Mode::Reference;
            else if (arg == "--auto") mode = Mode::Auto;
            else if (arg == "--verify") verify = true;
            else throw std::runtime_error("unknown argument: " + arg);
        }

        const auto capture = qw3::read_gdn_capture(capture_path);
        const bool use_metal = mode == Mode::Metal || (mode == Mode::Auto && qw3::gdn_metal_available());
        auto summaries = use_metal
            ? summarize_metal(capture, block_tokens)
            : qw3::gdn_summarize_capture_reference(capture, block_tokens);

        if (verify && use_metal) {
            const auto reference = qw3::gdn_summarize_capture_reference(capture, block_tokens);
            if (reference.size() != summaries.size()) throw std::runtime_error("summary block count mismatch");
            double worst_t = 0.0, worst_z = 0.0;
            for (size_t i = 0; i < summaries.size(); ++i) {
                worst_t = std::max(worst_t, max_abs(summaries[i].T, reference[i].T));
                worst_z = std::max(worst_z, max_abs(summaries[i].Z, reference[i].Z));
            }
            std::cout << "verify max_abs T=" << worst_t << " Z=" << worst_z << "\n";
            if (worst_t > 5e-4 || worst_z > 5e-4) {
                throw std::runtime_error("Metal affine summary exceeds fp64 qualification tolerance");
            }
        }

        qw3::write_gdn_summary_archive(output_path, summaries);
        std::cout << "wrote " << summaries.size() << " block summaries to " << output_path
                  << " using " << (use_metal ? "Metal" : "fp64 reference") << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "qw3-gdn-summarize: " << e.what() << "\n";
        return 1;
    }
}
