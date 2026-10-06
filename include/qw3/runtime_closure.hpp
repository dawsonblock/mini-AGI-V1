#pragma once

#include <string>

namespace qw3 {

struct RuntimeClosureMeasurements {
    std::string foundation_digest;
    std::string tokenizer_digest;
    std::string runtime_binary_digest;
    std::string kvmem_archive_root;
    std::string adapter_set_root;
    std::string retrieval_policy_root;
    std::string skill_policy_root;
    std::string artifact_root;
    std::string closure_digest;
    bool verified = false;
};

std::string current_executable_path();
std::string runtime_path_sha256_v154(const std::string &path);

// RuntimeClosure2 / ArtifactClosure1 helpers.  For GGUF models the tokenizer is
// physically embedded in the model, so its conservative identity is the GGUF
// byte digest.  For HF model directories this hashes exactly the tokenizer
// inputs QW3 reads: tokenizer.json, config.json, and tokenizer_config.json when
// present (otherwise generation_config.json when present).
std::string tokenizer_identity_sha256_v155(const std::string &model_path);
std::string physical_artifact_sha256_v155(const std::string &path);

std::string served_artifact_root_v154(
    const std::string &foundation_digest,
    const std::string &tokenizer_digest,
    const std::string &kv_archive_root,
    const std::string &adapter_set_root,
    const std::string &retrieval_policy_root,
    const std::string &skill_policy_root,
    const std::string &runtime_binary_digest,
    const std::string &native_adapter_bundle_root);

std::string runtime_closure_digest_v154(
    const std::string &epoch_id,
    const std::string &manifest_digest,
    const std::string &artifact_root,
    const std::string &foundation_digest,
    const std::string &runtime_binary_digest,
    const std::string &kvmem_archive_root,
    const std::string &adapter_set_root,
    const std::string &native_adapter_bundle_root);

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
    const std::string &native_adapter_bundle_root);

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
    const std::string &native_adapter_bundle_root);

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
    const std::string &native_adapter_bundle_root);

} // namespace qw3
