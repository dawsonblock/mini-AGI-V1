#include "qw3/runtime_closure.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

int main() {
    try {
        const std::string root = qw3::served_artifact_root_v154(
            std::string(64, '1'), std::string(64, '2'), std::string(64, '0'),
            std::string(64, '4'), std::string(64, '5'), std::string(64, '6'),
            std::string(64, '7'), std::string(64, '8'));
        if (root != "48c09aa70f043158301a54928550724290af03889e698ae5931a04f8ac593868") {
            throw std::runtime_error("served artifact root does not match Python canonical hash");
        }
        const std::string closure = qw3::runtime_closure_digest_v154(
            "sha256:" + std::string(64, 'a'), std::string(64, 'b'),
            std::string(64, 'c'), std::string(64, '1'), std::string(64, '7'),
            std::string(64, '0'), std::string(64, '4'), std::string(64, '8'));
        if (closure != "c26c4810a3a8ad256c6efb3bbae7e15fb241db3192c5aae5889295e8eaab234b") {
            throw std::runtime_error("runtime closure digest does not match Python canonical hash");
        }
        const auto tmp = std::filesystem::temp_directory_path() / "qw3-runtime-closure-tree";
        std::filesystem::remove_all(tmp);
        std::filesystem::create_directories(tmp / "sub");
        { std::ofstream a(tmp / "a.txt", std::ios::binary); a << "A"; }
        { std::ofstream b(tmp / "sub/b.bin", std::ios::binary); b << "BC"; }
        const std::string tree = qw3::runtime_path_sha256_v154(tmp.string());
        std::filesystem::remove_all(tmp);
        if (tree != "c36e8f47adf9a053fad7795d89b75e1c3c9a634b99141423331f9d8d100913c2") {
            throw std::runtime_error("runtime tree digest does not match Python canonical hash");
        }
        std::cout << "runtime closure canonical parity passed\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
