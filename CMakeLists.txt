cmake_minimum_required(VERSION 3.16)

project(qw3 LANGUAGES CXX)

find_package(Threads REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(QW3_BUILD_TESTS "Build QW3 host and CUDA tests" ON)
option(QW3_ENABLE_CUDA "Build CUDA native kernels" OFF)
option(QW3_ENABLE_FLASHINFER "Build optional FlashInfer prefill adapter" OFF)
option(QW3_ENABLE_MACOS_HOST "Enable macOS host-only portability profile" ON)
option(QW3_ENABLE_METAL_GDN "Build Apple Metal Gated-DeltaNet qualification kernel" ON)

if(APPLE AND QW3_ENABLE_CUDA)
    message(FATAL_ERROR "QW3 native CUDA kernels are not available on macOS. Build host utilities with QW3_ENABLE_CUDA=OFF and use macos/qw3-mac with llama.cpp/Metal or MLX for generation.")
endif()
set(QW3_GDN_SM120_AOT_DIR "" CACHE PATH
    "Directory containing generated gdn_sm120_0.h and gdn_sm120_0.o")
set(QW3_RMS_FP4_SM120_AOT_DIR "" CACHE PATH
    "Directory containing generated rms_fp4_sm120_0.h and rms_fp4_sm120_0.o")
set(QW3_CUTE_DSL_RUNTIME_DIR "" CACHE PATH
    "Directory containing libcute_dsl_runtime.so for SM120 CuTe DSL AOT objects")
set(QW3_FLASHINFER_INCLUDE_DIR "" CACHE PATH "FlashInfer include directory (e.g. /path/to/flashinfer/include)")
set(QW3_FLASHINFER_CUTLASS_INCLUDE_DIR "" CACHE PATH "CUTLASS include directory used by FlashInfer")
set(QW3_NVFP4_TEST_MODEL "" CACHE PATH "Optional local NVFP4 model for checkpoint parity testing")

if(QW3_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    # Default to Ampere+ — INT8 MMA m16n8k32.s8.s8.s32 (mmq_q8.cu) needs
    # sm_80. Override with -DCMAKE_CUDA_ARCHITECTURES=...
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 80)
    endif()
    set(QW3_CUDA_SRCS src/kernels_cuda.cu src/gated_delta_net.cu src/mmvq_q8.cu src/fattn_vec_decode.cu src/mmq_q8.cu src/kernel_legacy.cu)
    set(QW3_BUILD_NVFP4 OFF)
    set(QW3_FLASHINFER_SM120_SWAP_AB OFF)
    if(QW3_ENABLE_FLASHINFER)
        if(NOT QW3_FLASHINFER_INCLUDE_DIR)
            message(FATAL_ERROR "QW3_ENABLE_FLASHINFER=ON requires -DQW3_FLASHINFER_INCLUDE_DIR=/path/to/flashinfer/include")
        endif()
        get_filename_component(QW3_FLASHINFER_DATA_DIR
            "${QW3_FLASHINFER_INCLUDE_DIR}" DIRECTORY)
        set(QW3_FLASHINFER_CCCL_INCLUDE_DIRS)
        foreach(QW3_FLASHINFER_CCCL_DIR
                "${QW3_FLASHINFER_DATA_DIR}/cccl/cub"
                "${QW3_FLASHINFER_DATA_DIR}/cccl/libcudacxx/include"
                "${QW3_FLASHINFER_DATA_DIR}/cccl/thrust")
            if(EXISTS "${QW3_FLASHINFER_CCCL_DIR}")
                list(APPEND QW3_FLASHINFER_CCCL_INCLUDE_DIRS
                    "${QW3_FLASHINFER_CCCL_DIR}")
            endif()
        endforeach()
        if(QW3_FLASHINFER_CCCL_INCLUDE_DIRS)
            message(STATUS "Using FlashInfer bundled CCCL headers")
        endif()
        if("${CMAKE_CUDA_ARCHITECTURES}" MATCHES "(^|;)120a(-real|-virtual)?(;|$)")
            set(QW3_BUILD_NVFP4 ON)
            if(NOT QW3_FLASHINFER_CUTLASS_INCLUDE_DIR)
                set(QW3_FLASHINFER_CUTLASS_INCLUDE_DIR
                    "${QW3_FLASHINFER_DATA_DIR}/cutlass/include"
                    CACHE PATH "CUTLASS include directory used by FlashInfer" FORCE)
            endif()
            if(NOT EXISTS "${QW3_FLASHINFER_CUTLASS_INCLUDE_DIR}/cutlass/arch/arch.h")
                message(FATAL_ERROR
                    "FlashInfer CUTLASS headers not found under ${QW3_FLASHINFER_CUTLASS_INCLUDE_DIR}; "
                    "set -DQW3_FLASHINFER_CUTLASS_INCLUDE_DIR=/path/to/cutlass/include")
            endif()
            get_filename_component(QW3_FLASHINFER_CUTLASS_DIR
                "${QW3_FLASHINFER_CUTLASS_INCLUDE_DIR}" DIRECTORY)
            set(QW3_FLASHINFER_CUTLASS_UTIL_INCLUDE_DIR
                "${QW3_FLASHINFER_CUTLASS_DIR}/tools/util/include")
            if(NOT EXISTS "${QW3_FLASHINFER_CUTLASS_UTIL_INCLUDE_DIR}/cutlass/util/command_line.h")
                message(FATAL_ERROR
                    "FlashInfer CUTLASS utility headers not found under "
                    "${QW3_FLASHINFER_CUTLASS_UTIL_INCLUDE_DIR}")
            endif()
            set(QW3_FLASHINFER_SM120_HEADER
                "${QW3_FLASHINFER_INCLUDE_DIR}/flashinfer/gemm/fp4_gemm_cutlass_template_sm120.h")
            file(READ "${QW3_FLASHINFER_SM120_HEADER}" QW3_FLASHINFER_SM120_HEADER_CONTENT)
            string(FIND "${QW3_FLASHINFER_SM120_HEADER_CONTENT}"
                "bool SwapAB" QW3_FLASHINFER_SM120_SWAP_AB_INDEX)
            if(NOT QW3_FLASHINFER_SM120_SWAP_AB_INDEX EQUAL -1)
                set(QW3_FLASHINFER_SM120_SWAP_AB ON)
                message(STATUS "FlashInfer SM120 SwapAB API detected")
            endif()
            list(APPEND QW3_CUDA_SRCS
                src/nvfp4_linear_adapter.cu
                src/fp8_scaled_mm_adapter.cu)
            message(STATUS "QW3 native NVFP4 enabled for ${CMAKE_CUDA_ARCHITECTURES}")
        endif()
        list(APPEND QW3_CUDA_SRCS
            src/flashinfer_prefill_adapter.cu
            src/flashinfer_decode_adapter.cu
            src/flashinfer_vision_adapter.cu)
    endif()
else()
    if(QW3_ENABLE_FLASHINFER)
        message(FATAL_ERROR "QW3_ENABLE_FLASHINFER requires QW3_ENABLE_CUDA=ON")
    endif()
    set(QW3_CUDA_SRCS src/kernels_cuda_stub.cpp)
endif()

if(APPLE AND QW3_ENABLE_METAL_GDN)
    enable_language(OBJCXX)
    set(QW3_METAL_GDN_SRCS src/gdn_metal.mm)
else()
    set(QW3_METAL_GDN_SRCS src/gdn_metal_stub.cpp)
endif()

add_library(qw3_core
    src/engine.cpp
    src/gguf.cpp
    src/model_source.cpp
    ${QW3_CUDA_SRCS}
    src/qwen_native.cpp
    src/qwen_native_backend.cpp
    src/qwen_executor.cpp
    src/qwen_weights.cpp
    src/qwen_config.cpp
    src/kvmem_store.cpp
    src/kvmem_coherent_state.cpp
    src/gdn_reference.cpp
    src/gdn_segment_io.cpp
    src/gdn_summary_runtime.cpp
    src/gdn_runtime_bundle.cpp
    ${QW3_METAL_GDN_SRCS}
    src/kvmem_archive.cpp
    src/native_lora.cpp
    src/runtime_closure.cpp
    src/anthropic_adapter.cpp
    src/vision_cpu_frontend.cpp
    src/vision_gpu_frontend.cpp
    src/harness_semantics.cpp
    src/tool_call_stream.cpp
    src/tokenizer.cpp
)

target_include_directories(qw3_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_include_directories(qw3_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

target_compile_options(qw3_core PRIVATE
    -Wall -Wextra -Wpedantic
)
target_compile_definitions(qw3_core PRIVATE
    QW3_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)
target_link_libraries(qw3_core PUBLIC Threads::Threads)

if(APPLE AND QW3_ENABLE_MACOS_HOST)
    target_compile_definitions(qw3_core PRIVATE QW3_PLATFORM_MACOS=1)
    # Accelerate is linked for a stable Apple-native math dependency boundary
    # even though generation is delegated to llama.cpp/Metal or MLX today.
    find_library(QW3_ACCELERATE_FRAMEWORK Accelerate)
    if(QW3_ACCELERATE_FRAMEWORK)
        target_link_libraries(qw3_core PUBLIC "${QW3_ACCELERATE_FRAMEWORK}")
    endif()

if(APPLE AND QW3_ENABLE_METAL_GDN)
    find_library(QW3_METAL_FRAMEWORK Metal REQUIRED)
    find_library(QW3_FOUNDATION_FRAMEWORK Foundation REQUIRED)
    target_compile_definitions(qw3_core PRIVATE QW3_PLATFORM_METAL_GDN=1)
    target_link_libraries(qw3_core PUBLIC "${QW3_METAL_FRAMEWORK}" "${QW3_FOUNDATION_FRAMEWORK}")
    set_source_files_properties(src/gdn_metal.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
endif()
endif()

# Archive manifests identify the complete GGUF by SHA-256. OpenSSL's optimized
# implementation makes the first uncached multi-GiB scan much faster; the
# source retains a portable implementation for minimal deployments.
find_package(OpenSSL QUIET COMPONENTS Crypto)
if(OpenSSL_FOUND)
    target_compile_definitions(qw3_core PRIVATE QW3_HAVE_OPENSSL=1)
    target_link_libraries(qw3_core PUBLIC OpenSSL::Crypto)
endif()

if(QW3_ENABLE_CUDA)
    target_compile_definitions(qw3_core PRIVATE QW3_ENABLE_CUDA=1)
    target_link_libraries(qw3_core PUBLIC CUDA::cudart CUDA::cublas CUDA::cublasLt)
    set_property(TARGET qw3_core PROPERTY CUDA_STANDARD 17)
    set_property(TARGET qw3_core PROPERTY CUDA_STANDARD_REQUIRED ON)
    if(QW3_ENABLE_FLASHINFER)
        target_compile_definitions(qw3_core PRIVATE QW3_ENABLE_FLASHINFER=1)
        target_include_directories(qw3_core SYSTEM PRIVATE
            ${QW3_FLASHINFER_INCLUDE_DIR}
            ${QW3_FLASHINFER_CCCL_INCLUDE_DIRS})
    endif()
    if(QW3_BUILD_NVFP4)
        target_compile_definitions(qw3_core PRIVATE QW3_ENABLE_NVFP4=1)
        target_compile_options(qw3_core PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
        if(QW3_FLASHINFER_SM120_SWAP_AB)
            target_compile_definitions(qw3_core PRIVATE
                QW3_FLASHINFER_SM120_SWAP_AB=1)
        endif()
        target_include_directories(qw3_core SYSTEM PRIVATE
            ${QW3_FLASHINFER_CUTLASS_INCLUDE_DIR}
            ${QW3_FLASHINFER_CUTLASS_UTIL_INCLUDE_DIR})
    endif()
    if(QW3_CUDA_VERBOSE)
        target_compile_options(qw3_core PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>)
    endif()
    if(QW3_GDN_SM120_AOT_DIR)
        if(NOT EXISTS "${QW3_GDN_SM120_AOT_DIR}/gdn_sm120_0.h" OR
           NOT EXISTS "${QW3_GDN_SM120_AOT_DIR}/gdn_sm120_0.o")
            message(FATAL_ERROR
                "QW3_GDN_SM120_AOT_DIR must contain gdn_sm120_0.h and gdn_sm120_0.o")
        endif()
        if(NOT QW3_CUTE_DSL_RUNTIME_DIR)
            message(FATAL_ERROR
                "QW3_GDN_SM120_AOT_DIR requires QW3_CUTE_DSL_RUNTIME_DIR")
        endif()
        find_library(QW3_CUTE_DSL_RUNTIME_LIBRARY
            NAMES cute_dsl_runtime
            PATHS "${QW3_CUTE_DSL_RUNTIME_DIR}"
            NO_DEFAULT_PATH
            REQUIRED)
        set_source_files_properties(
            "${QW3_GDN_SM120_AOT_DIR}/gdn_sm120_0.o"
            PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
        target_sources(qw3_core PRIVATE
            src/gdn_sm120_aot_adapter.cpp
            "${QW3_GDN_SM120_AOT_DIR}/gdn_sm120_0.o")
        target_include_directories(qw3_core PRIVATE
            "${QW3_GDN_SM120_AOT_DIR}")
        target_compile_definitions(qw3_core PRIVATE QW3_ENABLE_GDN_SM120_AOT=1)
        target_link_libraries(qw3_core PUBLIC
            "${QW3_CUTE_DSL_RUNTIME_LIBRARY}")
        target_link_options(qw3_core PUBLIC
            "-Wl,-rpath,${QW3_CUTE_DSL_RUNTIME_DIR}")
        message(STATUS "QW3 SM120 GDN AOT backend enabled")
    endif()
    if(QW3_RMS_FP4_SM120_AOT_DIR)
        if(NOT EXISTS "${QW3_RMS_FP4_SM120_AOT_DIR}/rms_fp4_sm120_0.h" OR
           NOT EXISTS "${QW3_RMS_FP4_SM120_AOT_DIR}/rms_fp4_sm120_0.o")
            message(FATAL_ERROR
                "QW3_RMS_FP4_SM120_AOT_DIR must contain "
                "rms_fp4_sm120_0.h and rms_fp4_sm120_0.o")
        endif()
        if(NOT QW3_CUTE_DSL_RUNTIME_DIR)
            message(FATAL_ERROR
                "QW3_RMS_FP4_SM120_AOT_DIR requires QW3_CUTE_DSL_RUNTIME_DIR")
        endif()
        find_library(QW3_CUTE_DSL_RUNTIME_LIBRARY
            NAMES cute_dsl_runtime
            PATHS "${QW3_CUTE_DSL_RUNTIME_DIR}"
            NO_DEFAULT_PATH
            REQUIRED)
        set_source_files_properties(
            "${QW3_RMS_FP4_SM120_AOT_DIR}/rms_fp4_sm120_0.o"
            PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
        target_sources(qw3_core PRIVATE
            src/rms_fp4_sm120_aot_adapter.cpp
            "${QW3_RMS_FP4_SM120_AOT_DIR}/rms_fp4_sm120_0.o")
        target_include_directories(qw3_core PRIVATE
            "${QW3_RMS_FP4_SM120_AOT_DIR}")
        target_compile_definitions(qw3_core PRIVATE
            QW3_ENABLE_RMS_FP4_SM120_AOT=1)
        target_link_libraries(qw3_core PUBLIC
            "${QW3_CUTE_DSL_RUNTIME_LIBRARY}")
        target_link_options(qw3_core PUBLIC
            "-Wl,-rpath,${QW3_CUTE_DSL_RUNTIME_DIR}")
        message(STATUS "QW3 SM120 RMSNorm+NVFP4 AOT backend enabled")
    endif()
endif()

add_executable(qw3 src/qw3_cli.cpp src/qw3_server.cpp)
target_include_directories(qw3 SYSTEM PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
target_compile_definitions(qw3 PRIVATE
    QW3_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)
target_link_libraries(qw3 PRIVATE qw3_core)

add_executable(qw3-inspect src/qw3_inspect.cpp)
target_link_libraries(qw3-inspect PRIVATE qw3_core)

add_executable(qw3-gdn-summarize src/gdn_summary_cli.cpp)
target_link_libraries(qw3-gdn-summarize PRIVATE qw3_core)

add_executable(qw3-gdn-runtime-cli src/gdn_runtime_cli.cpp)
target_link_libraries(qw3-gdn-runtime-cli PRIVATE qw3_core)
set_target_properties(qw3-gdn-runtime-cli PROPERTIES OUTPUT_NAME qw3-gdn-runtime)

if(QW3_BUILD_TESTS)
    add_executable(qw3-prompt-render tests/prompt_render_test.cpp)
    target_link_libraries(qw3-prompt-render PRIVATE qw3_core)
    enable_testing()
    add_test(NAME qw3-prompt-render COMMAND qw3-prompt-render)

    add_executable(qw3-kvmem-store tests/kvmem_store_test.cpp)
    target_link_libraries(qw3-kvmem-store PRIVATE qw3_core)
    add_test(NAME qw3-kvmem-store COMMAND qw3-kvmem-store)

    add_executable(qw3-kvmem-coherent-state tests/kvmem_coherent_state_test.cpp)
    target_link_libraries(qw3-kvmem-coherent-state PRIVATE qw3_core)
    add_test(NAME qw3-kvmem-coherent-state COMMAND qw3-kvmem-coherent-state)

    add_executable(qw3-gdn-reference tests/gdn_reference_test.cpp)
    target_link_libraries(qw3-gdn-reference PRIVATE qw3_core)
    add_test(NAME qw3-gdn-reference COMMAND qw3-gdn-reference)

    add_executable(qw3-gdn-segment-io tests/gdn_segment_io_test.cpp)
    target_link_libraries(qw3-gdn-segment-io PRIVATE qw3_core)
    add_test(NAME qw3-gdn-segment-io COMMAND qw3-gdn-segment-io)

    add_executable(qw3-gdn-metal-interface tests/gdn_metal_interface_test.cpp)
    target_link_libraries(qw3-gdn-metal-interface PRIVATE qw3_core)
    add_test(NAME qw3-gdn-metal-interface COMMAND qw3-gdn-metal-interface)

    add_executable(qw3-gdn-summary-runtime tests/gdn_summary_runtime_test.cpp)
    target_link_libraries(qw3-gdn-summary-runtime PRIVATE qw3_core)
    add_test(NAME qw3-gdn-summary-runtime COMMAND qw3-gdn-summary-runtime)

    add_executable(qw3-gdn-runtime-bundle tests/gdn_runtime_bundle_test.cpp)
    target_link_libraries(qw3-gdn-runtime-bundle PRIVATE qw3_core)
    add_test(NAME qw3-gdn-runtime-bundle COMMAND qw3-gdn-runtime-bundle)

    add_executable(qw3-kvmem-prefix-reuse-policy
        tests/kvmem_prefix_reuse_policy_test.cpp)
    target_include_directories(qw3-kvmem-prefix-reuse-policy PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    add_test(NAME qw3-kvmem-prefix-reuse-policy
        COMMAND qw3-kvmem-prefix-reuse-policy)

    add_executable(qw3-kvmem-refresh-policy
        tests/kvmem_refresh_policy_test.cpp)
    target_include_directories(qw3-kvmem-refresh-policy PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    add_test(NAME qw3-kvmem-refresh-policy
        COMMAND qw3-kvmem-refresh-policy)

    add_executable(qw3-kvmem-request-plan
        tests/kvmem_request_plan_test.cpp)
    target_include_directories(qw3-kvmem-request-plan PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/include)
    add_test(NAME qw3-kvmem-request-plan
        COMMAND qw3-kvmem-request-plan)

    add_executable(qw3-kvmem-session-manager
        tests/kvmem_session_manager_test.cpp)
    target_include_directories(qw3-kvmem-session-manager PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/include)
    add_test(NAME qw3-kvmem-session-manager
        COMMAND qw3-kvmem-session-manager)

    add_executable(qw3-kvmem-executor-scheduler
        tests/kvmem_executor_scheduler_test.cpp)
    target_include_directories(qw3-kvmem-executor-scheduler PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/include)
    add_test(NAME qw3-kvmem-executor-scheduler
        COMMAND qw3-kvmem-executor-scheduler)

    add_executable(qw3-kvmem-executor-slot-runtime
        tests/kvmem_executor_slot_runtime_test.cpp)
    target_include_directories(qw3-kvmem-executor-slot-runtime PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/include)
    add_test(NAME qw3-kvmem-executor-slot-runtime
        COMMAND qw3-kvmem-executor-slot-runtime)


    add_executable(qw3-kvmem-physical-executor-pool
        tests/kvmem_physical_executor_pool_test.cpp)
    target_include_directories(qw3-kvmem-physical-executor-pool PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/include)
    add_test(NAME qw3-kvmem-physical-executor-pool
        COMMAND qw3-kvmem-physical-executor-pool)

    add_executable(qw3-kvmem-resource-admission
        tests/kvmem_resource_admission_test.cpp)
    target_include_directories(qw3-kvmem-resource-admission PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/include)
    add_test(NAME qw3-kvmem-resource-admission
        COMMAND qw3-kvmem-resource-admission)

    add_executable(qw3-kvmem-session-snapshot
        tests/kvmem_session_snapshot_test.cpp)
    target_include_directories(qw3-kvmem-session-snapshot PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/include)
    add_test(NAME qw3-kvmem-session-snapshot
        COMMAND qw3-kvmem-session-snapshot)

    add_executable(qw3-kvmem-runtime-profile
        tests/kvmem_runtime_profile_test.cpp)
    target_include_directories(qw3-kvmem-runtime-profile PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/include)
    add_test(NAME qw3-kvmem-runtime-profile
        COMMAND qw3-kvmem-runtime-profile)

    add_executable(qw3-tool-call-stream tests/tool_call_stream_test.cpp)
    target_include_directories(qw3-tool-call-stream PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(qw3-tool-call-stream PRIVATE qw3_core)
    add_test(NAME qw3-tool-call-stream COMMAND qw3-tool-call-stream)

    add_executable(qw3-anthropic-adapter tests/anthropic_adapter_test.cpp)
    target_include_directories(qw3-anthropic-adapter PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_include_directories(qw3-anthropic-adapter SYSTEM PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
    target_link_libraries(qw3-anthropic-adapter PRIVATE qw3_core)
    add_test(NAME qw3-anthropic-adapter COMMAND qw3-anthropic-adapter)

    add_executable(qw3-harness-semantics tests/harness_semantics_test.cpp)
    target_include_directories(qw3-harness-semantics PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(qw3-harness-semantics PRIVATE qw3_core)
    add_test(NAME qw3-harness-semantics COMMAND qw3-harness-semantics)

    add_executable(qw3-tokenizer-pretokenizer
        tests/tokenizer_pretokenizer_test.cpp)
    target_include_directories(qw3-tokenizer-pretokenizer PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(qw3-tokenizer-pretokenizer PRIVATE qw3_core)
    add_test(NAME qw3-tokenizer-pretokenizer
        COMMAND qw3-tokenizer-pretokenizer)

    add_executable(qw3-pinned-kv-tier tests/pinned_kv_tier_test.cpp)
    target_link_libraries(qw3-pinned-kv-tier PRIVATE qw3_core)
    add_test(NAME qw3-pinned-kv-tier COMMAND qw3-pinned-kv-tier)

    add_executable(qw3-nvme-kv-tier tests/nvme_kv_tier_test.cpp)
    target_link_libraries(qw3-nvme-kv-tier PRIVATE qw3_core)
    add_test(NAME qw3-nvme-kv-tier COMMAND qw3-nvme-kv-tier)

    add_executable(qw3-kvmem-archive tests/kvmem_archive_test.cpp)
    target_link_libraries(qw3-kvmem-archive PRIVATE qw3_core)
    add_test(NAME qw3-kvmem-archive COMMAND qw3-kvmem-archive)

    add_executable(qw3-native-lora tests/native_lora_test.cpp)
    target_include_directories(qw3-native-lora SYSTEM PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
    target_link_libraries(qw3-native-lora PRIVATE qw3_core)
    add_test(NAME qw3-native-lora COMMAND qw3-native-lora)

    add_executable(qw3-runtime-closure tests/runtime_closure_test.cpp)
    target_include_directories(qw3-runtime-closure SYSTEM PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
    target_link_libraries(qw3-runtime-closure PRIVATE qw3_core)
    add_test(NAME qw3-runtime-closure COMMAND qw3-runtime-closure)

    add_executable(qw3-runtime-closure-v155 tests/runtime_closure_v155_test.cpp)
    target_include_directories(qw3-runtime-closure-v155 SYSTEM PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
    target_link_libraries(qw3-runtime-closure-v155 PRIVATE qw3_core)
    add_test(NAME qw3-runtime-closure-v155 COMMAND qw3-runtime-closure-v155)

    if(QW3_ENABLE_CUDA)
        add_executable(qw3-vision-gpu-parity tests/vision_gpu_parity.cpp)
        target_include_directories(qw3-vision-gpu-parity PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src)
        target_link_libraries(qw3-vision-gpu-parity PRIVATE qw3_core)

        add_executable(qw3-mmq-parity tests/mmq_parity.cu)
        target_link_libraries(qw3-mmq-parity PRIVATE qw3_core)
        set_property(TARGET qw3-mmq-parity PROPERTY CUDA_STANDARD 17)
        set_property(TARGET qw3-mmq-parity PROPERTY CUDA_STANDARD_REQUIRED ON)
        add_test(NAME qw3-mmq-parity COMMAND qw3-mmq-parity)

        add_executable(qw3-rope-remap-parity tests/rope_remap_parity.cu)
        target_link_libraries(qw3-rope-remap-parity PRIVATE qw3_core)
        set_property(TARGET qw3-rope-remap-parity PROPERTY CUDA_STANDARD 17)
        set_property(TARGET qw3-rope-remap-parity PROPERTY CUDA_STANDARD_REQUIRED ON)
        add_test(NAME qw3-rope-remap-parity COMMAND qw3-rope-remap-parity)

        add_executable(qw3-kvmem-assembly tests/kvmem_assembly.cu)
        target_link_libraries(qw3-kvmem-assembly PRIVATE qw3_core)
        set_property(TARGET qw3-kvmem-assembly PROPERTY CUDA_STANDARD 17)
        set_property(TARGET qw3-kvmem-assembly PROPERTY CUDA_STANDARD_REQUIRED ON)
        add_test(NAME qw3-kvmem-assembly COMMAND qw3-kvmem-assembly)

        add_executable(qw3-kvmem-immutable-k tests/kvmem_immutable_k.cu)
        target_link_libraries(qw3-kvmem-immutable-k PRIVATE qw3_core)
        set_property(TARGET qw3-kvmem-immutable-k PROPERTY CUDA_STANDARD 17)
        set_property(TARGET qw3-kvmem-immutable-k PROPERTY CUDA_STANDARD_REQUIRED ON)
        add_test(NAME qw3-kvmem-immutable-k COMMAND qw3-kvmem-immutable-k)

        add_executable(qw3-rope-remap-batched-parity tests/rope_remap_batched_parity.cu)
        target_link_libraries(qw3-rope-remap-batched-parity PRIVATE qw3_core)
        set_property(TARGET qw3-rope-remap-batched-parity PROPERTY CUDA_STANDARD 17)
        set_property(TARGET qw3-rope-remap-batched-parity PROPERTY CUDA_STANDARD_REQUIRED ON)
        add_test(NAME qw3-rope-remap-batched-parity COMMAND qw3-rope-remap-batched-parity)

        add_executable(qw3-rope-remap-drift tests/rope_remap_drift.cu)
        target_link_libraries(qw3-rope-remap-drift PRIVATE qw3_core)
        set_property(TARGET qw3-rope-remap-drift PROPERTY CUDA_STANDARD 17)
        set_property(TARGET qw3-rope-remap-drift PROPERTY CUDA_STANDARD_REQUIRED ON)

        add_executable(qw3-kvmem-deltanet-score tests/kvmem_deltanet_score.cu)
        target_link_libraries(qw3-kvmem-deltanet-score PRIVATE qw3_core)
        set_property(TARGET qw3-kvmem-deltanet-score PROPERTY CUDA_STANDARD 17)
        set_property(TARGET qw3-kvmem-deltanet-score PROPERTY CUDA_STANDARD_REQUIRED ON)
        add_test(NAME qw3-kvmem-deltanet-score COMMAND qw3-kvmem-deltanet-score)

        add_executable(qw3-kvmem-softmax-pages tests/kvmem_softmax_pages.cu)
        target_link_libraries(qw3-kvmem-softmax-pages PRIVATE qw3_core)
        set_property(TARGET qw3-kvmem-softmax-pages PROPERTY CUDA_STANDARD 17)
        set_property(TARGET qw3-kvmem-softmax-pages PROPERTY CUDA_STANDARD_REQUIRED ON)
        add_test(NAME qw3-kvmem-softmax-pages COMMAND qw3-kvmem-softmax-pages)

        add_executable(qw3-kvmem-kmean-merge tests/kvmem_kmean_merge.cu)
        target_link_libraries(qw3-kvmem-kmean-merge PRIVATE qw3_core)
        set_property(TARGET qw3-kvmem-kmean-merge PROPERTY CUDA_STANDARD 17)
        set_property(TARGET qw3-kvmem-kmean-merge PROPERTY CUDA_STANDARD_REQUIRED ON)
        add_test(NAME qw3-kvmem-kmean-merge COMMAND qw3-kvmem-kmean-merge)

        add_executable(qw3-kvmem-exactmass-largeblocks tests/kvmem_exactmass_largeblocks.cu)
        target_link_libraries(qw3-kvmem-exactmass-largeblocks PRIVATE qw3_core)
        set_property(TARGET qw3-kvmem-exactmass-largeblocks PROPERTY CUDA_STANDARD 17)
        set_property(TARGET qw3-kvmem-exactmass-largeblocks PROPERTY CUDA_STANDARD_REQUIRED ON)
        add_test(NAME qw3-kvmem-exactmass-largeblocks COMMAND qw3-kvmem-exactmass-largeblocks)
        if(QW3_BUILD_NVFP4)
            add_executable(qw3-mixed-linear-parity tests/mixed_linear_parity.cu)
            target_link_libraries(qw3-mixed-linear-parity PRIVATE qw3_core)
            set_property(TARGET qw3-mixed-linear-parity PROPERTY CUDA_STANDARD 17)
            set_property(TARGET qw3-mixed-linear-parity PROPERTY CUDA_STANDARD_REQUIRED ON)
            add_test(NAME qw3-mixed-linear-parity COMMAND qw3-mixed-linear-parity)
            add_executable(qw3-nvfp4-decode-bench tests/nvfp4_decode_bench.cu)
            target_link_libraries(qw3-nvfp4-decode-bench PRIVATE qw3_core)
            set_property(TARGET qw3-nvfp4-decode-bench PROPERTY CUDA_STANDARD 17)
            set_property(TARGET qw3-nvfp4-decode-bench PROPERTY CUDA_STANDARD_REQUIRED ON)
            if(QW3_NVFP4_TEST_MODEL)
                add_test(NAME qw3-nvfp4-checkpoint-parity
                    COMMAND qw3-mixed-linear-parity ${QW3_NVFP4_TEST_MODEL})
            endif()
        endif()
    endif()
endif()

# Test assertions must remain effective in Release configurations too.
if(QW3_BUILD_TESTS)
    get_property(QW3_VALIDATION_TARGETS DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
    foreach(QW3_VALIDATION_TARGET IN LISTS QW3_VALIDATION_TARGETS)
        get_target_property(QW3_VALIDATION_SOURCES ${QW3_VALIDATION_TARGET} SOURCES)
        if("${QW3_VALIDATION_SOURCES}" MATCHES "(^|;)tests/")
            target_compile_options(${QW3_VALIDATION_TARGET} PRIVATE
                $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-UNDEBUG>
                $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/UNDEBUG>)
        endif()
    endforeach()
endif()
