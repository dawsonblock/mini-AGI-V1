#include "qw3/runtime_closure.hpp"

#include "qw3/kvmem_archive.hpp"
#include "json.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <unistd.h>

#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif

namespace qw3 {
namespace {
using nlohmann::json;

constexpr char kZeroRoot[] =
    "0000000000000000000000000000000000000000000000000000000000000000";

bool is_hex64(const std::string &v) {
    return v.size() == 64 && std::all_of(v.begin(), v.end(), [](unsigned char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    });
}

void require_hex64(const std::string &v, const char *name) {
    if (!is_hex64(v)) {
        throw std::runtime_error(std::string("runtime closure ") + name +
                                 " must be lowercase sha256 hex");
    }
}

std::string canonical_sha256(const json &value) {
    return kvmem_archive_sha256_text(value.dump());
}

} // namespace

std::string current_executable_path() {
#if defined(__linux__)
    std::array<char, 4096> buf{};
    const ssize_t n = ::readlink("/proc/self/exe", buf.data(), buf.size() - 1);
    if (n <= 0 || static_cast<size_t>(n) >= buf.size()) {
        throw std::runtime_error(std::string("cannot resolve /proc/self/exe: ") +
                                 std::strerror(errno));
    }
    buf[static_cast<size_t>(n)] = '\0';
    return std::filesystem::weakly_canonical(buf.data()).string();
#elif defined(__APPLE__)
    uint32_t size = 0;
    (void)_NSGetExecutablePath(nullptr, &size);
    if (size == 0) throw std::runtime_error("cannot size executable path");
    std::string path(size, '\0');
    if (_NSGetExecutablePath(path.data(), &size) != 0) {
        throw std::runtime_error("cannot resolve executable path");
    }
    path.resize(std::strlen(path.c_str()));
    return std::filesystem::weakly_canonical(path).string();
#else
    throw std::runtime_error("RuntimeClosure1 cannot resolve the current executable on this platform");
#endif
}

std::string runtime_path_sha256_v154(const std::string &path) {
    namespace fs = std::filesystem;
    const fs::path root = fs::weakly_canonical(fs::path(path));
    if (fs::is_symlink(fs::symlink_status(root))) {
        throw std::runtime_error("runtime identity path cannot be a symlink");
    }
    if (fs::is_regular_file(root)) return kvmem_archive_model_sha256(root.string());
    if (!fs::is_directory(root)) {
        throw std::runtime_error("runtime identity path is neither file nor directory: " + path);
    }
    std::vector<fs::path> files;
    for (const auto &entry : fs::recursive_directory_iterator(root)) {
        const auto st = entry.symlink_status();
        if (fs::is_symlink(st)) {
            throw std::runtime_error("runtime identity directory contains symlink: " + entry.path().string());
        }
        if (fs::is_regular_file(st)) files.push_back(entry.path());
    }
    std::sort(files.begin(), files.end(), [&](const fs::path &a, const fs::path &b) {
        return fs::relative(a, root).generic_string() < fs::relative(b, root).generic_string();
    });
    std::string canonical = "mini-agi-v15.4-runtime-tree-v1\n";
    for (const auto &file : files) {
        const std::string rel = fs::relative(file, root).generic_string();
        canonical += rel;
        canonical.push_back('\0');
        canonical += std::to_string(static_cast<unsigned long long>(fs::file_size(file)));
        canonical.push_back('\0');
        canonical += kvmem_archive_model_sha256(file.string());
        canonical.push_back('\n');
    }
    return kvmem_archive_sha256_text(canonical);
}

std::string served_artifact_root_v154(
    const std::string &foundation_digest,
    const std::string &tokenizer_digest,
    const std::string &kv_archive_root,
    const std::string &adapter_set_root,
    const std::string &retrieval_policy_root,
    const std::string &skill_policy_root,
    const std::string &runtime_binary_digest,
    const std::string &native_adapter_bundle_root) {
    const std::array<std::pair<const char *, const std::string *>, 8> roots{{
        {"foundation_digest", &foundation_digest},
        {"tokenizer_digest", &tokenizer_digest},
        {"kv_archive_root", &kv_archive_root},
        {"adapter_set_root", &adapter_set_root},
        {"retrieval_policy_root", &retrieval_policy_root},
        {"skill_policy_root", &skill_policy_root},
        {"runtime_binary_digest", &runtime_binary_digest},
        {"native_adapter_bundle_root", &native_adapter_bundle_root},
    }};
    for (const auto &pair : roots) require_hex64(*pair.second, pair.first);
    json body = {
        {"schema", "mini-agi-v15-served-artifact-root-v1"},
        {"foundation_digest", foundation_digest},
        {"tokenizer_digest", tokenizer_digest},
        {"kv_archive_root", kv_archive_root},
        {"adapter_set_root", adapter_set_root},
        {"retrieval_policy_root", retrieval_policy_root},
        {"skill_policy_root", skill_policy_root},
        {"runtime_binary_digest", runtime_binary_digest},
    };
    if (native_adapter_bundle_root != kZeroRoot) {
        body["native_adapter_bundle_root"] = native_adapter_bundle_root;
    }
    return canonical_sha256(body);
}

std::string runtime_closure_digest_v154(
    const std::string &epoch_id,
    const std::string &manifest_digest,
    const std::string &artifact_root,
    const std::string &foundation_digest,
    const std::string &runtime_binary_digest,
    const std::string &kvmem_archive_root,
    const std::string &adapter_set_root,
    const std::string &native_adapter_bundle_root) {
    json body = {
        {"schema", "mini-agi-v15.4-runtime-closure-v1"},
        {"epoch_id", epoch_id},
        {"manifest_digest", manifest_digest},
        {"artifact_root", artifact_root},
        {"foundation_digest", foundation_digest},
        {"runtime_binary_digest", runtime_binary_digest},
        {"kvmem_archive_root", kvmem_archive_root},
        {"adapter_set_root", adapter_set_root},
        {"native_adapter_bundle_root", native_adapter_bundle_root},
    };
    return canonical_sha256(body);
}

RuntimeClosureMeasurements verify_runtime_closure_v154(
    const std::string &model_path,
    const std::string &kvmem_archive_dir,
    const std::string &epoch_id,
    const std::string &manifest_digest,
    const std::string &expected_artifact_root,
    const std::string &expected_foundation_digest,
    const std::string &tokenizer_digest,
    const std::string &expected_kvmem_archive_root,
    const std::string &adapter_set_root,
    const std::string &retrieval_policy_root,
    const std::string &skill_policy_root,
    const std::string &expected_runtime_binary_digest,
    const std::string &native_adapter_bundle_root) {
    const std::array<std::pair<const char *, const std::string *>, 9> roots{{
        {"artifact_root", &expected_artifact_root},
        {"foundation_digest", &expected_foundation_digest},
        {"tokenizer_digest", &tokenizer_digest},
        {"kvmem_archive_root", &expected_kvmem_archive_root},
        {"adapter_set_root", &adapter_set_root},
        {"retrieval_policy_root", &retrieval_policy_root},
        {"skill_policy_root", &skill_policy_root},
        {"runtime_binary_digest", &expected_runtime_binary_digest},
        {"native_adapter_bundle_root", &native_adapter_bundle_root},
    }};
    for (const auto &pair : roots) require_hex64(*pair.second, pair.first);
    RuntimeClosureMeasurements out;
    out.foundation_digest = runtime_path_sha256_v154(model_path);
    if (out.foundation_digest != expected_foundation_digest) {
        throw std::runtime_error("governed runtime model bytes do not match foundation_digest");
    }
    out.runtime_binary_digest =
        kvmem_archive_model_sha256(current_executable_path());
    if (out.runtime_binary_digest != expected_runtime_binary_digest) {
        throw std::runtime_error("governed runtime executable bytes do not match runtime_binary_digest");
    }
    if (kvmem_archive_dir.empty()) {
        out.kvmem_archive_root = kZeroRoot;
    } else {
        out.kvmem_archive_root = kvmem_archive_model_sha256(
            (std::filesystem::path(kvmem_archive_dir) / "manifest.json").string());
    }
    if (out.kvmem_archive_root != expected_kvmem_archive_root) {
        throw std::runtime_error("governed KVMem manifest bytes do not match kv_archive_root");
    }
    out.artifact_root = served_artifact_root_v154(
        expected_foundation_digest, tokenizer_digest, expected_kvmem_archive_root,
        adapter_set_root, retrieval_policy_root, skill_policy_root,
        expected_runtime_binary_digest, native_adapter_bundle_root);
    if (out.artifact_root != expected_artifact_root) {
        throw std::runtime_error("served artifact root is inconsistent with component roots");
    }
    out.closure_digest = runtime_closure_digest_v154(
        epoch_id, manifest_digest, expected_artifact_root, out.foundation_digest,
        out.runtime_binary_digest, out.kvmem_archive_root, adapter_set_root,
        native_adapter_bundle_root);
    out.verified = true;
    return out;
}


std::string physical_artifact_sha256_v155(const std::string &path) {
    if (path.empty()) {
        throw std::runtime_error("physical artifact path is empty");
    }
    return runtime_path_sha256_v154(path);
}

std::string tokenizer_identity_sha256_v155(const std::string &model_path) {
    namespace fs = std::filesystem;
    const fs::path root = fs::weakly_canonical(fs::path(model_path));
    if (fs::is_symlink(fs::symlink_status(root))) {
        throw std::runtime_error("tokenizer identity source cannot be a symlink");
    }
    if (fs::is_regular_file(root)) {
        // GGUF carries tokenizer.ggml.* metadata inside the same physical file.
        // Hashing the complete GGUF is deliberately conservative: any tokenizer
        // or model byte mutation invalidates the tokenizer identity.
        return kvmem_archive_model_sha256(root.string());
    }
    if (!fs::is_directory(root)) {
        throw std::runtime_error("tokenizer identity source is neither GGUF nor HF directory");
    }

    std::vector<fs::path> files;
    const fs::path tokenizer = root / "tokenizer.json";
    const fs::path config = root / "config.json";
    if (!fs::is_regular_file(tokenizer) || fs::is_symlink(fs::symlink_status(tokenizer))) {
        throw std::runtime_error("HF tokenizer identity requires regular tokenizer.json");
    }
    if (!fs::is_regular_file(config) || fs::is_symlink(fs::symlink_status(config))) {
        throw std::runtime_error("HF tokenizer identity requires regular config.json");
    }
    files.push_back(tokenizer);
    files.push_back(config);
    const fs::path tok_cfg = root / "tokenizer_config.json";
    const fs::path gen_cfg = root / "generation_config.json";
    if (fs::exists(tok_cfg)) {
        if (!fs::is_regular_file(tok_cfg) || fs::is_symlink(fs::symlink_status(tok_cfg))) {
            throw std::runtime_error("unsafe tokenizer_config.json");
        }
        files.push_back(tok_cfg);
    } else if (fs::exists(gen_cfg)) {
        if (!fs::is_regular_file(gen_cfg) || fs::is_symlink(fs::symlink_status(gen_cfg))) {
            throw std::runtime_error("unsafe generation_config.json");
        }
        files.push_back(gen_cfg);
    }
    std::sort(files.begin(), files.end(), [&](const fs::path &a, const fs::path &b) {
        return fs::relative(a, root).generic_string() < fs::relative(b, root).generic_string();
    });
    std::string canonical = "mini-agi-v15.5-tokenizer-identity-v1\n";
    for (const auto &file : files) {
        const std::string rel = fs::relative(file, root).generic_string();
        canonical += rel;
        canonical.push_back('\0');
        canonical += std::to_string(static_cast<unsigned long long>(fs::file_size(file)));
        canonical.push_back('\0');
        canonical += kvmem_archive_model_sha256(file.string());
        canonical.push_back('\n');
    }
    return kvmem_archive_sha256_text(canonical);
}

std::string runtime_closure_digest_v155(
    const std::string &epoch_id,
    const std::string &manifest_digest,
    const std::string &artifact_root,
    const std::string &foundation_digest,
    const std::string &tokenizer_digest,
    const std::string &runtime_binary_digest,
    const std::string &kvmem_archive_root,
    const std::string &adapter_set_root,
    const std::string &retrieval_policy_root,
    const std::string &skill_policy_root,
    const std::string &native_adapter_bundle_root) {
    json body = {
        {"schema", "mini-agi-v15.5-runtime-closure-v1"},
        {"epoch_id", epoch_id},
        {"manifest_digest", manifest_digest},
        {"artifact_root", artifact_root},
        {"foundation_digest", foundation_digest},
        {"tokenizer_digest", tokenizer_digest},
        {"runtime_binary_digest", runtime_binary_digest},
        {"kvmem_archive_root", kvmem_archive_root},
        {"adapter_set_root", adapter_set_root},
        {"retrieval_policy_root", retrieval_policy_root},
        {"skill_policy_root", skill_policy_root},
        {"native_adapter_bundle_root", native_adapter_bundle_root},
    };
    return canonical_sha256(body);
}

namespace {
std::string measure_optional_root_v155(
        const std::string &path,
        const std::string &expected_root,
        const char *name) {
    require_hex64(expected_root, name);
    if (expected_root == kZeroRoot) {
        if (!path.empty()) {
            throw std::runtime_error(std::string(name) +
                " is zero-root but a physical artifact path was supplied");
        }
        return kZeroRoot;
    }
    if (path.empty()) {
        throw std::runtime_error(std::string(name) +
            " is nonzero but no physical artifact path was supplied");
    }
    const std::string measured = physical_artifact_sha256_v155(path);
    if (measured != expected_root) {
        throw std::runtime_error(std::string("physical ") + name +
            " bytes do not match authorized root");
    }
    return measured;
}
} // namespace

RuntimeClosureMeasurements verify_runtime_closure_v155(
    const std::string &model_path,
    const std::string &kvmem_archive_dir,
    const std::string &adapter_set_artifact,
    const std::string &retrieval_policy_artifact,
    const std::string &skill_policy_artifact,
    const std::string &epoch_id,
    const std::string &manifest_digest,
    const std::string &expected_artifact_root,
    const std::string &expected_foundation_digest,
    const std::string &expected_tokenizer_digest,
    const std::string &expected_kvmem_archive_root,
    const std::string &expected_adapter_set_root,
    const std::string &expected_retrieval_policy_root,
    const std::string &expected_skill_policy_root,
    const std::string &expected_runtime_binary_digest,
    const std::string &native_adapter_bundle_root) {
    const std::array<std::pair<const char *, const std::string *>, 9> roots{{
        {"artifact_root", &expected_artifact_root},
        {"foundation_digest", &expected_foundation_digest},
        {"tokenizer_digest", &expected_tokenizer_digest},
        {"kvmem_archive_root", &expected_kvmem_archive_root},
        {"adapter_set_root", &expected_adapter_set_root},
        {"retrieval_policy_root", &expected_retrieval_policy_root},
        {"skill_policy_root", &expected_skill_policy_root},
        {"runtime_binary_digest", &expected_runtime_binary_digest},
        {"native_adapter_bundle_root", &native_adapter_bundle_root},
    }};
    for (const auto &pair : roots) require_hex64(*pair.second, pair.first);

    RuntimeClosureMeasurements out;
    out.foundation_digest = runtime_path_sha256_v154(model_path);
    if (out.foundation_digest != expected_foundation_digest) {
        throw std::runtime_error("governed runtime model bytes do not match foundation_digest");
    }
    out.tokenizer_digest = tokenizer_identity_sha256_v155(model_path);
    if (out.tokenizer_digest != expected_tokenizer_digest) {
        throw std::runtime_error("loaded tokenizer inputs do not match tokenizer_digest");
    }
    out.runtime_binary_digest = kvmem_archive_model_sha256(current_executable_path());
    if (out.runtime_binary_digest != expected_runtime_binary_digest) {
        throw std::runtime_error("governed runtime executable bytes do not match runtime_binary_digest");
    }
    if (kvmem_archive_dir.empty()) {
        out.kvmem_archive_root = kZeroRoot;
    } else {
        const std::filesystem::path manifest =
            std::filesystem::path(kvmem_archive_dir) / "manifest.json";
        if (!std::filesystem::is_regular_file(manifest) ||
            std::filesystem::is_symlink(std::filesystem::symlink_status(manifest))) {
            throw std::runtime_error("KVMem archive manifest is missing or unsafe");
        }
        out.kvmem_archive_root = kvmem_archive_model_sha256(manifest.string());
    }
    if (out.kvmem_archive_root != expected_kvmem_archive_root) {
        throw std::runtime_error("governed KVMem manifest bytes do not match kv_archive_root");
    }
    out.adapter_set_root = measure_optional_root_v155(
        adapter_set_artifact, expected_adapter_set_root, "adapter_set_root");
    out.retrieval_policy_root = measure_optional_root_v155(
        retrieval_policy_artifact, expected_retrieval_policy_root, "retrieval_policy_root");
    out.skill_policy_root = measure_optional_root_v155(
        skill_policy_artifact, expected_skill_policy_root, "skill_policy_root");

    out.artifact_root = served_artifact_root_v154(
        out.foundation_digest, out.tokenizer_digest, out.kvmem_archive_root,
        out.adapter_set_root, out.retrieval_policy_root, out.skill_policy_root,
        out.runtime_binary_digest, native_adapter_bundle_root);
    if (out.artifact_root != expected_artifact_root) {
        throw std::runtime_error("served artifact root is inconsistent with measured component roots");
    }
    out.closure_digest = runtime_closure_digest_v155(
        epoch_id, manifest_digest, out.artifact_root, out.foundation_digest,
        out.tokenizer_digest, out.runtime_binary_digest, out.kvmem_archive_root,
        out.adapter_set_root, out.retrieval_policy_root, out.skill_policy_root,
        native_adapter_bundle_root);
    out.verified = true;
    return out;
}

} // namespace qw3
