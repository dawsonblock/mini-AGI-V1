#include "qw3/runtime_closure.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

int main() {
    try {
        namespace fs = std::filesystem;
        const auto tmp = fs::temp_directory_path() / "qw3-runtime-closure-v155";
        fs::remove_all(tmp);
        fs::create_directories(tmp / "model");
        {
            std::ofstream f(tmp / "model/tokenizer.json", std::ios::binary);
            f << "{\"model\":{\"vocab\":{\"a\":0},\"merges\":[]}}";
        }
        {
            std::ofstream f(tmp / "model/config.json", std::ios::binary);
            f << "{\"text_config\":{\"bos_token_id\":0,\"eos_token_id\":0}}";
        }
        {
            std::ofstream f(tmp / "model/tokenizer_config.json", std::ios::binary);
            f << "{\"eos_token\":\"a\"}";
        }
        const std::string tokenizer = qw3::tokenizer_identity_sha256_v155((tmp / "model").string());
        if (tokenizer != "e08eb1c23144f6e3783f130c2df4168e00401e9422929daa17d4041deac1039f") {
            throw std::runtime_error("v15.5 tokenizer identity does not match Python parity value");
        }

        const auto policy = tmp / "policy.bin";
        { std::ofstream f(policy, std::ios::binary); f << "physical-policy"; }
        const std::string physical = qw3::physical_artifact_sha256_v155(policy.string());
        if (physical != "a1a028016bbaee1b6921de3990aab412551edc9135bf268a4d7ebf4b23bcbe2b") {
            throw std::runtime_error("physical artifact SHA-256 mismatch");
        }

        const std::string closure = qw3::runtime_closure_digest_v155(
            "sha256:" + std::string(64, 'a'), std::string(64, 'b'),
            std::string(64, 'c'), std::string(64, '1'), std::string(64, '2'),
            std::string(64, '7'), std::string(64, '0'), std::string(64, '4'),
            std::string(64, '5'), std::string(64, '6'), std::string(64, '8'));
        if (closure != "f95c0f6af564dcd2a6235dbc625ea6663d0d97a0f60f31f6e4ffa2978ad4f4b2") {
            throw std::runtime_error("v15.5 runtime closure digest does not match Python parity value");
        }

        fs::remove_all(tmp);
        std::cout << "runtime closure v15.5 physical parity passed\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
