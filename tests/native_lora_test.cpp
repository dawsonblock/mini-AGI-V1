#include "qw3/kvmem_archive.hpp"
#include "qw3/native_lora.hpp"

#include "json.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

static void write_f32(const fs::path &path, const std::vector<float> &values) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot create test payload");
    out.write(reinterpret_cast<const char *>(values.data()),
              static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!out) throw std::runtime_error("cannot write test payload");
}

static void require_close(float actual, float expected, const char *label) {
    if (std::fabs(actual - expected) > 1e-6f) {
        throw std::runtime_error(std::string(label) + " mismatch");
    }
}

int main() {
    const fs::path root = fs::temp_directory_path() / "qw3-native-lora-test";
    fs::remove_all(root);
    fs::create_directories(root);
    try {
        // A: [rank=2, in=3], B: [out=4, rank=2]
        const std::vector<float> a{1.f, 0.f, 0.f,
                                   0.f, 1.f, 0.f};
        const std::vector<float> b{1.f, 0.f,
                                   0.f, 1.f,
                                   1.f, 1.f,
                                  -1.f, 2.f};
        const fs::path ap = root / "A.f32";
        const fs::path bp = root / "B.f32";
        write_f32(ap, a);
        write_f32(bp, b);
        const std::string asha = qw3::kvmem_archive_model_sha256(ap.string());
        const std::string bsha = qw3::kvmem_archive_model_sha256(bp.string());

        const std::string adapter_root(64, 'a');
        const std::string model_sha(64, 'b');
        const std::string governance = "sha256:" + std::string(64, 'c');
        const std::string qualified = "sha256:" + std::string(64, 'd');

        json manifest = {
            {"schema", "qw3-native-lora-bundle-v1"},
            {"adapter_set_root", adapter_root},
            {"model_sha256", model_sha},
            {"foundation_model_digest", governance},
            {"qualified_candidate_digest", qualified},
            {"tensors", json::array({{
                {"target", "output.weight"},
                {"rank", 2},
                {"in_features", 3},
                {"out_features", 4},
                {"scale", 0.5},
                {"a", {{"file", "A.f32"}, {"sha256", asha}, {"bytes", a.size() * sizeof(float)}}},
                {"b", {{"file", "B.f32"}, {"sha256", bsha}, {"bytes", b.size() * sizeof(float)}}}
            }})}
        };
        const fs::path mp = root / "manifest.json";
        {
            std::ofstream out(mp, std::ios::trunc);
            out << manifest.dump();
        }
        const std::string bundle_root = qw3::kvmem_archive_model_sha256(mp.string());
        const auto bundle = qw3::NativeLoraHostBundle::load(
            root.string(), adapter_root, bundle_root, model_sha, 3, 4);
        if (bundle.info().tensor_count != 1 || bundle.info().max_rank != 2) {
            throw std::runtime_error("native LoRA bundle metadata mismatch");
        }
        std::vector<float> output(4, 0.0f);
        bundle.apply_output_cpu({2.f, 4.f, 8.f}, output);
        require_close(output[0], 1.f, "output[0]");
        require_close(output[1], 2.f, "output[1]");
        require_close(output[2], 3.f, "output[2]");
        require_close(output[3], 3.f, "output[3]");

        // Byte-level tampering must be detected before use.
        auto tampered = b;
        tampered[0] += 1.f;
        write_f32(bp, tampered);
        bool rejected = false;
        try {
            (void)qw3::NativeLoraHostBundle::load(
                root.string(), adapter_root, bundle_root, model_sha, 3, 4);
        } catch (const std::exception &) {
            rejected = true;
        }
        if (!rejected) throw std::runtime_error("tampered LoRA payload was accepted");

        fs::remove_all(root);
        std::cout << "native LoRA host bundle test passed\n";
        return 0;
    } catch (const std::exception &e) {
        fs::remove_all(root);
        std::cerr << e.what() << "\n";
        return 1;
    }
}
