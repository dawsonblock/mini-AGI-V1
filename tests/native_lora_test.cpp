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

        // ---- v2 bundle: per-layer attention projections -------------
        const fs::path root2 = fs::temp_directory_path() / "qw3-native-lora-v2-test";
        fs::remove_all(root2);
        fs::create_directories(root2);

        // Model geometry: n_layers=2, hidden=3, q_rows=4, kv_rows=2.
        // Layer-0 q_proj entry: A [1,3], B [4,1], scale 2.
        const std::vector<float> qa{1.f, -1.f, 0.5f};
        const std::vector<float> qb{1.f, 2.f, -1.f, 0.f};
        const fs::path qap = root2 / "qA.f32";
        const fs::path qbp = root2 / "qB.f32";
        write_f32(qap, qa);
        write_f32(qbp, qb);
        const std::string qasha = qw3::kvmem_archive_model_sha256(qap.string());
        const std::string qbsha = qw3::kvmem_archive_model_sha256(qbp.string());

        qw3::NativeLoraAttentionDims dims;
        dims.n_layers = 2;
        dims.hidden = 3;
        dims.q_rows = 4;
        dims.kv_rows = 2;
        dims.o_in = 4;  // post-attention mid width feeding o_proj

        json manifest2 = {
            {"schema", "qw3-native-lora-bundle-v2"},
            {"adapter_set_root", adapter_root},
            {"model_sha256", model_sha},
            {"foundation_model_digest", governance},
            {"qualified_candidate_digest", qualified},
            {"tensors", json::array({{
                {"target", "self_attn.q_proj"},
                {"layer", 0},
                {"rank", 1},
                {"in_features", 3},
                {"out_features", 4},
                {"scale", 2.0},
                {"a", {{"file", "qA.f32"}, {"sha256", qasha}, {"bytes", qa.size() * sizeof(float)}}},
                {"b", {{"file", "qB.f32"}, {"sha256", qbsha}, {"bytes", qb.size() * sizeof(float)}}}
            }})}
        };
        const fs::path mp2 = root2 / "manifest.json";
        { std::ofstream out(mp2, std::ios::trunc); out << manifest2.dump(); }
        const std::string bundle2_root = qw3::kvmem_archive_model_sha256(mp2.string());
        const auto bundle2 = qw3::NativeLoraHostBundle::load(
            root2.string(), adapter_root, bundle2_root, model_sha, 3, 4,
            &dims);
        if (!bundle2.has_attention()) {
            throw std::runtime_error("v2 bundle must report attention coverage");
        }
        // oracle: q_out += 2 * qb * (qa . input)
        // input=(2,4,8): qa.x = 1*2 + -1*4 + 0.5*8 = 2; scaled 2 -> 4
        std::vector<float> qout(4, 0.0f);
        bundle2.apply_projection_cpu(qw3::NativeLoraKind::Q, 0,
                                     {2.f, 4.f, 8.f}, qout);
        require_close(qout[0], 4.f, "q[0]");
        require_close(qout[1], 8.f, "q[1]");
        require_close(qout[2], -4.f, "q[2]");
        require_close(qout[3], 0.f, "q[3]");
        // no entry on layer 1 -> no-op
        std::vector<float> qout1(4, 1.0f);
        bundle2.apply_projection_cpu(qw3::NativeLoraKind::Q, 1,
                                     {2.f, 4.f, 8.f}, qout1);
        require_close(qout1[0], 1.f, "q_l1[0]");

        // v2 dims mismatch must fail closed
        {
            json bad = manifest2;
            bad["tensors"][0]["in_features"] = 9;
            std::ofstream out(mp2, std::ios::trunc); out << bad.dump();
            bool dim_rejected = false;
            try {
                (void)qw3::NativeLoraHostBundle::load(
                    root2.string(), adapter_root,
                    qw3::kvmem_archive_model_sha256(mp2.string()),
                    model_sha, 3, 4, &dims);
            } catch (const std::exception &) { dim_rejected = true; }
            if (!dim_rejected) throw std::runtime_error("v2 dims mismatch accepted");
        }
        // layer index out of range must fail closed
        {
            json bad = manifest2;
            bad["tensors"][0]["layer"] = 7;
            std::ofstream out(mp2, std::ios::trunc); out << bad.dump();
            bool layer_rejected = false;
            try {
                (void)qw3::NativeLoraHostBundle::load(
                    root2.string(), adapter_root,
                    qw3::kvmem_archive_model_sha256(mp2.string()),
                    model_sha, 3, 4, &dims);
            } catch (const std::exception &) { layer_rejected = true; }
            if (!layer_rejected) throw std::runtime_error("v2 bad layer accepted");
        }
        // attention entries under a v1 schema must be rejected
        {
            json bad = manifest2;
            bad["schema"] = "qw3-native-lora-bundle-v1";
            std::ofstream out(mp2, std::ios::trunc); out << bad.dump();
            bool schema_rejected = false;
            try {
                (void)qw3::NativeLoraHostBundle::load(
                    root2.string(), adapter_root,
                    qw3::kvmem_archive_model_sha256(mp2.string()),
                    model_sha, 3, 4, &dims);
            } catch (const std::exception &) { schema_rejected = true; }
            if (!schema_rejected) throw std::runtime_error("attention target accepted under v1 schema");
        }
        fs::remove_all(root2);

        fs::remove_all(root);
        std::cout << "native LoRA host bundle test passed\n";
        return 0;
    } catch (const std::exception &e) {
        fs::remove_all(root);
        std::cerr << e.what() << "\n";
        return 1;
    }
}
