#include "qw3/gdn_metal.hpp"
#include "qw3/gdn_runtime_bundle.hpp"
#include "qw3/gdn_segment_io.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<uint32_t> parse_ids(const std::string& s) {
    std::vector<uint32_t> out;
    std::stringstream ss(s);
    std::string part;
    while (std::getline(ss, part, ',')) {
        if (part.empty()) throw std::runtime_error("empty block id");
        const unsigned long v = std::stoul(part);
        if (v > 0xffffffffUL) throw std::runtime_error("block id out of range");
        out.push_back(static_cast<uint32_t>(v));
    }
    if (out.empty()) throw std::runtime_error("--blocks requires at least one id");
    return out;
}

void set_identity_field(qw3::KvMemCacheIdentity& id, const std::string& flag,
                        const std::string& value) {
    if (flag == "--base-model-digest") id.base_model_digest = value;
    else if (flag == "--adapter-set-digest") id.adapter_set_digest = value;
    else if (flag == "--tokenizer-digest") id.tokenizer_digest = value;
    else if (flag == "--layer-layout-digest") id.layer_layout_digest = value;
    else if (flag == "--position-scheme") id.position_scheme = value;
    else if (flag == "--recurrence-impl") id.recurrence_impl = value;
    else throw std::runtime_error("unknown identity flag: " + flag);
}

bool is_identity_flag(const std::string& a) {
    return a == "--base-model-digest" || a == "--adapter-set-digest" ||
           a == "--tokenizer-digest" || a == "--layer-layout-digest" ||
           a == "--position-scheme" || a == "--recurrence-impl";
}

void write_state(const std::filesystem::path& path, const std::vector<float>& state) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot open state output: " + path.string());
    out.write(reinterpret_cast<const char*>(state.data()),
              static_cast<std::streamsize>(state.size() * sizeof(float)));
    if (!out) throw std::runtime_error("failed writing state output: " + path.string());
}

std::vector<const qw3::GdnBlockAffineSummary*> selected_layer_summaries(
    const qw3::GdnRuntimeBundle& bundle,
    uint32_t layer,
    const std::vector<uint32_t>& ids) {
    std::vector<const qw3::GdnBlockAffineSummary*> all;
    for (const auto& s : bundle.summaries) {
        if (s.layer_index == layer) all.push_back(&s);
    }
    std::sort(all.begin(), all.end(), [](const auto* a, const auto* b) {
        if (a->position_start != b->position_start) return a->position_start < b->position_start;
        return a->token_count < b->token_count;
    });
    std::vector<const qw3::GdnBlockAffineSummary*> out;
    out.reserve(ids.size());
    for (uint32_t id : ids) {
        if (id >= all.size()) throw std::runtime_error("selected block id exceeds layer geometry");
        out.push_back(all[id]);
    }
    return out;
}

qw3::GdnMetalSummaryChainInputs make_metal_chain(
    const std::vector<const qw3::GdnBlockAffineSummary*>& selected) {
    if (selected.empty()) throw std::runtime_error("cannot build an empty Metal summary chain");
    const auto& first = *selected.front();
    qw3::GdnMetalSummaryChainInputs in;
    in.block_count = static_cast<uint32_t>(selected.size());
    in.num_v_heads = first.num_v_heads;
    in.head_dim = first.head_dim;
    const size_t per = static_cast<size_t>(in.num_v_heads) * in.head_dim * in.head_dim;
    in.T.reserve(per * selected.size());
    in.Z.reserve(per * selected.size());
    in.state.assign(per, 0.0f);
    for (const auto* s : selected) {
        if (s->num_v_heads != in.num_v_heads || s->head_dim != in.head_dim) {
            throw std::runtime_error("Metal summary chain geometry mismatch");
        }
        in.T.insert(in.T.end(), s->T.begin(), s->T.end());
        in.Z.insert(in.Z.end(), s->Z.begin(), s->Z.end());
    }
    in.validate();
    return in;
}

void usage() {
    std::cerr <<
        "qw3-gdn-runtime bundle <out.qgdb> --model-id ID "
        "--base-model-digest X --adapter-set-digest X --tokenizer-digest X "
        "--layer-layout-digest X --position-scheme X --recurrence-impl X "
        "<layer.qgds>...\n"
        "qw3-gdn-runtime inspect <bundle.qgdb>\n"
        "qw3-gdn-runtime reconstruct <bundle.qgdb> --blocks 0,2,4 --out-dir DIR "
        "[--metal] [--verify] [--verify-tol EPS] [identity override flags]\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 3) { usage(); return 2; }
        const std::string cmd = argv[1];
        if (cmd == "bundle") {
            const std::string out_path = argv[2];
            std::string model_id;
            qw3::KvMemCacheIdentity identity;
            std::vector<std::string> archives;
            for (int i = 3; i < argc; ++i) {
                const std::string a = argv[i];
                if (a == "--model-id") {
                    if (++i >= argc) throw std::runtime_error("--model-id requires a value");
                    model_id = argv[i];
                } else if (is_identity_flag(a)) {
                    if (++i >= argc) throw std::runtime_error(a + " requires a value");
                    set_identity_field(identity, a, argv[i]);
                } else if (a.rfind("--", 0) == 0) {
                    throw std::runtime_error("unknown argument: " + a);
                } else {
                    archives.push_back(a);
                }
            }
            if (model_id.empty()) throw std::runtime_error("--model-id is required");
            if (archives.empty()) throw std::runtime_error("at least one .qgds archive is required");
            std::vector<qw3::GdnBlockAffineSummary> summaries;
            for (const auto& path : archives) {
                auto v = qw3::read_gdn_summary_archive(path);
                summaries.insert(summaries.end(), v.begin(), v.end());
            }
            for (const auto& s : summaries) {
                if (s.model_id != model_id) {
                    throw std::runtime_error("summary model_id does not match --model-id");
                }
            }
            auto bundle = qw3::build_gdn_runtime_bundle(identity, summaries);
            qw3::write_gdn_runtime_bundle(out_path, bundle);
            std::cout << "wrote=" << out_path
                      << " layers=" << bundle.layer_indices().size()
                      << " blocks=" << bundle.block_count()
                      << " summaries=" << bundle.summaries.size() << "\n";
            return 0;
        }
        if (cmd == "inspect") {
            if (argc != 3) { usage(); return 2; }
            const auto b = qw3::read_gdn_runtime_bundle(argv[2]);
            const auto layers = b.layer_indices();
            const auto blocks = qw3::gdn_runtime_blocks(b);
            bool conv_aware = !b.summaries.empty() && b.summaries.front().version >= 3;
            std::cout << "version=" << b.version << " model_id=" << b.model_id
                      << " layers=" << layers.size() << " blocks=" << blocks.size()
                      << " summaries=" << b.summaries.size()
                      << " conv_state=" << (conv_aware ? "available" : "legacy-missing") << "\n";
            std::cout << "identity base=" << b.identity.base_model_digest
                      << " adapters=" << b.identity.adapter_set_digest
                      << " tokenizer=" << b.identity.tokenizer_digest
                      << " layout=" << b.identity.layer_layout_digest
                      << " position=" << b.identity.position_scheme
                      << " recurrence=" << b.identity.recurrence_impl << "\n";
            std::cout << "layer_indices=";
            for (size_t i = 0; i < layers.size(); ++i) {
                if (i) std::cout << ',';
                std::cout << layers[i];
            }
            std::cout << "\n";
            for (const auto& block : blocks) {
                std::cout << "block=" << block.block_id
                          << " position=" << block.position_start
                          << " tokens=" << block.token_count << "\n";
            }
            return 0;
        }
        if (cmd == "plan") {
            const auto bundle = qw3::read_gdn_runtime_bundle(argv[2]);
            std::vector<uint32_t> ids;
            uint32_t repair_tokens = 0;
            double max_fraction = 0.25;
            for (int i = 3; i < argc; ++i) {
                const std::string a = argv[i];
                if (a == "--blocks") {
                    if (++i >= argc) throw std::runtime_error("--blocks requires a value");
                    ids = parse_ids(argv[i]);
                } else if (a == "--repair-tokens") {
                    if (++i >= argc) throw std::runtime_error("--repair-tokens requires a value");
                    const unsigned long v = std::stoul(argv[i]);
                    if (v > 0xffffffffUL) throw std::runtime_error("--repair-tokens out of range");
                    repair_tokens = static_cast<uint32_t>(v);
                } else if (a == "--max-replay-fraction") {
                    if (++i >= argc) throw std::runtime_error("--max-replay-fraction requires a value");
                    max_fraction = std::stod(argv[i]);
                } else {
                    throw std::runtime_error("unknown argument: " + a);
                }
            }
            if (ids.empty()) throw std::runtime_error("--blocks is required");
            if (repair_tokens == 0) throw std::runtime_error("--repair-tokens must be > 0");
            const auto p = qw3::plan_gdn_seam_replay(bundle, ids, repair_tokens, max_fraction);
            std::cout << "selected_tokens=" << p.selected_tokens
                      << " discontinuities=" << p.discontinuities
                      << " conv_repair_tokens=" << p.conv_repair_tokens
                      << " hidden_repair_tokens=" << p.hidden_repair_tokens
                      << " replay_tokens=" << p.replay_tokens
                      << " replay_fraction=" << p.replay_fraction
                      << " full_model_boundary_exact=" << (p.full_model_boundary_exact ? "yes" : "no")
                      << " qualification_shortcut_eligible=" << (p.qualification_shortcut_eligible ? "yes" : "no")
                      << " parity_gate_required=" << (p.parity_gate_required ? "yes" : "no") << "\n";
            for (const auto& w : p.windows) {
                std::cout << "window block=" << w.downstream_block_id
                          << " compact_start=" << w.compact_token_start
                          << " tokens=" << w.token_count
                          << " source_start=" << w.source_position_start << "\n";
            }
            if (p.parity_gate_required) {
                std::cout << "note=sparse bounded replay is qualification-only; exact selected replay remains the fallback authority\n";
            }
            return 0;
        }
        if (cmd == "reconstruct") {
            const auto bundle = qw3::read_gdn_runtime_bundle(argv[2]);
            auto expected = bundle.identity;
            std::vector<uint32_t> ids;
            std::string out_dir;
            bool use_metal = false;
            bool verify = false;
            double verify_tol = 5e-4;
            for (int i = 3; i < argc; ++i) {
                const std::string a = argv[i];
                if (a == "--blocks") {
                    if (++i >= argc) throw std::runtime_error("--blocks requires a value");
                    ids = parse_ids(argv[i]);
                } else if (a == "--out-dir") {
                    if (++i >= argc) throw std::runtime_error("--out-dir requires a value");
                    out_dir = argv[i];
                } else if (a == "--metal") {
                    use_metal = true;
                } else if (a == "--verify") {
                    verify = true;
                    use_metal = true;
                } else if (a == "--verify-tol") {
                    if (++i >= argc) throw std::runtime_error("--verify-tol requires a value");
                    verify_tol = std::stod(argv[i]);
                    if (!(verify_tol > 0.0) || !std::isfinite(verify_tol)) {
                        throw std::runtime_error("--verify-tol must be finite and > 0");
                    }
                } else if (is_identity_flag(a)) {
                    if (++i >= argc) throw std::runtime_error(a + " requires a value");
                    set_identity_field(expected, a, argv[i]);
                } else {
                    throw std::runtime_error("unknown argument: " + a);
                }
            }
            if (ids.empty()) throw std::runtime_error("--blocks is required");
            if (out_dir.empty()) throw std::runtime_error("--out-dir is required");
            const auto cpu = qw3::reconstruct_gdn_runtime_bundle_cpu(bundle, ids, expected);
            std::filesystem::create_directories(out_dir);

            double worst_metal_cpu = 0.0;
            std::string metal_device;
            for (const auto& layer : cpu.layers) {
                std::vector<float> output_state = layer.state_column_major;
                if (use_metal) {
                    const auto selected = selected_layer_summaries(bundle, layer.layer_index, ids);
                    const auto metal = qw3::gdn_metal_apply_summary_chain(make_metal_chain(selected));
                    metal_device = metal.device_name;
                    if (metal.state.size() != layer.state_column_major.size()) {
                        throw std::runtime_error("Metal reconstruction state size mismatch");
                    }
                    for (size_t i = 0; i < metal.state.size(); ++i) {
                        worst_metal_cpu = std::max(
                            worst_metal_cpu,
                            std::abs(double(metal.state[i]) - double(layer.state_column_major[i])));
                    }
                    if (verify && worst_metal_cpu > verify_tol) {
                        throw std::runtime_error(
                            "Metal/CPU recurrent-state parity exceeded tolerance: " +
                            std::to_string(worst_metal_cpu) + " > " +
                            std::to_string(verify_tol));
                    }
                    output_state = metal.state;
                }
                const auto path = std::filesystem::path(out_dir) /
                    ("layer-" + std::to_string(layer.layer_index) + ".state.f32");
                write_state(path, output_state);
                if (!layer.conv_state_channel_major.empty()) {
                    const auto conv_path = std::filesystem::path(out_dir) /
                        ("layer-" + std::to_string(layer.layer_index) + ".conv.f32");
                    write_state(conv_path, layer.conv_state_channel_major);
                }
            }

            std::ofstream manifest(std::filesystem::path(out_dir) / "manifest.txt",
                                   std::ios::trunc);
            if (!manifest) throw std::runtime_error("cannot write reconstruction manifest");
            manifest << "model_id=" << bundle.model_id << "\n"
                     << "selected_blocks=";
            for (size_t i = 0; i < cpu.selected_block_ids.size(); ++i) {
                if (i) manifest << ',';
                manifest << cpu.selected_block_ids[i];
            }
            manifest << "\nstarts_at_zero=" << (cpu.starts_at_zero ? "true" : "false")
                     << "\ncontiguous=" << (cpu.contiguous ? "true" : "false")
                     << "\nconv_state_available=" << (cpu.conv_state_available ? "true" : "false")
                     << "\nfull_model_boundary_exact=" << (cpu.full_model_boundary_exact ? "true" : "false")
                     << "\nhidden_replay_required=" << (cpu.hidden_replay_required ? "true" : "false")
                     << "\nseam_repair_required=" << (cpu.seam_repair_required ? "true" : "false")
                     << "\nbackend=" << (use_metal ? "metal" : "cpu")
                     << "\n";
            if (use_metal) {
                manifest << "metal_device=" << metal_device << "\n"
                         << "metal_cpu_max_abs=" << worst_metal_cpu << "\n"
                         << "metal_verified=" << (verify ? "true" : "false") << "\n";
            }
            std::cout << "layers=" << cpu.layers.size()
                      << " selected_blocks=" << cpu.selected_block_ids.size()
                      << " backend=" << (use_metal ? "metal" : "cpu")
                      << " conv_state=" << (cpu.conv_state_available ? "yes" : "no")
                      << " boundary_exact=" << (cpu.full_model_boundary_exact ? "yes" : "no")
                      << " hidden_replay_required=" << (cpu.hidden_replay_required ? "yes" : "no");
            if (use_metal) std::cout << " metal_cpu_max_abs=" << worst_metal_cpu;
            std::cout << "\n";
            if (cpu.hidden_replay_required) {
                std::cout << "note=recurrent and Conv1D prepared-input boundary reconstructed; sparse hidden-state seam replay/parity is still required before serving activation\n";
            } else {
                std::cout << "note=gap-free prefix boundary reconstructs recurrent and Conv1D state exactly for the captured model identity\n";
            }
            return 0;
        }
        usage();
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "qw3-gdn-runtime: " << e.what() << "\n";
        return 1;
    }
}
