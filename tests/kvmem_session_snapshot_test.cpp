#include "kvmem_session_snapshot.hpp"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>

int main() {
    using qw3::detail::KvMemSessionManager;
    using qw3::detail::KvMemSessionSnapshotStore;

    const auto dir = std::filesystem::temp_directory_path() /
                     "qw3-kvmem-session-snapshot-test";
    std::filesystem::remove_all(dir);

    KvMemSessionManager::Record rec;
    rec.id = "session-a";
    rec.workspace_id = "workspace-a";
    rec.version = 7;
    rec.created_at = 100;
    rec.cold_rehydrates = 3;
    rec.tokens = {10, 20, 30, 40};
    rec.input_embedding_fingerprint = 999;
    rec.input_mrope_positions = {{{1,2,3}}, {{4,5,6}}};
    rec.pinned_token_spans.push_back(
        {1, 3, qw3::GenerationOptions::KvMemPinnedReason::RootTask});
    qw3::GenerationOptions::InputEmbeddingOverride emb;
    emb.token_id = 101;
    emb.source_token_id = 102;
    emb.position = {7,8,9};
    emb.embedding_row = 2;
    emb.embedding = {1.5f, 2.5f};
    rec.input_embedding_overrides.push_back(emb);

    KvMemSessionSnapshotStore store(dir.string(), "runtime:abc");
    const auto saved = store.save(rec);
    assert(saved.found);
    assert(saved.version == 7);
    assert(saved.token_count == 4);
    assert(saved.host_bytes == KvMemSessionManager::estimate_host_bytes(rec));
    assert(std::filesystem::exists(saved.path));

    qw3::KvMemSessionSnapshotInfo loaded_info;
    const auto loaded = store.load("session-a", &loaded_info);
    assert(loaded.id == rec.id);
    assert(loaded.workspace_id == rec.workspace_id);
    assert(loaded.version == rec.version);
    assert(loaded.tokens == rec.tokens);
    assert(loaded.input_embedding_fingerprint == rec.input_embedding_fingerprint);
    assert(loaded.input_embedding_overrides.size() == 1);
    assert(loaded.input_embedding_overrides[0].embedding == emb.embedding);
    assert(loaded.pinned_token_spans.size() == 1);
    assert(loaded_info.runtime_fingerprint == "runtime:abc");

    bool fingerprint_rejected = false;
    try {
        KvMemSessionSnapshotStore wrong(dir.string(), "runtime:different");
        (void)wrong.load("session-a");
    } catch (...) { fingerprint_rejected = true; }
    assert(fingerprint_rejected);

    // Corrupt one payload byte and prove integrity verification fails closed.
    {
        std::fstream f(saved.path, std::ios::binary | std::ios::in | std::ios::out);
        f.seekg(-1, std::ios::end);
        char c = 0;
        f.read(&c, 1);
        c ^= 0x5a;
        f.seekp(-1, std::ios::end);
        f.write(&c, 1);
    }
    bool corruption_rejected = false;
    try { (void)store.load("session-a"); }
    catch (...) { corruption_rejected = true; }
    assert(corruption_rejected);

    std::filesystem::remove_all(dir);
    std::cout << "kvmem_session_snapshot_test: PASS\n";
    return 0;
}
