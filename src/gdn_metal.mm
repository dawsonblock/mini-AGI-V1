#include "qw3/gdn_metal.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

#ifndef QW3_SOURCE_DIR
#define QW3_SOURCE_DIR "."
#endif

namespace qw3 {
namespace {

struct GdnDimsMetal {
    uint32_t T;
    uint32_t num_k_heads;
    uint32_t num_v_heads;
    uint32_t head_dim;
    uint32_t qkv_row_stride;
    uint32_t v_row_stride;
    uint32_t gb_row_stride;
    uint32_t out_row_stride;
};

void validate_finite(const std::vector<float>& values, const char* name) {
    for (float v : values) {
        if (!std::isfinite(v)) throw std::invalid_argument(std::string(name) + " contains non-finite value");
    }
}

id<MTLBuffer> shared_buffer(id<MTLDevice> device, const std::vector<float>& src) {
    return [device newBufferWithBytes:src.data()
                               length:src.size() * sizeof(float)
                              options:MTLResourceStorageModeShared];
}

id<MTLComputePipelineState> pipeline(id<MTLDevice> device, id<MTLLibrary> library, NSString* name) {
    id<MTLFunction> function = [library newFunctionWithName:name];
    if (function == nil) throw std::runtime_error("Metal function not found: " + std::string([name UTF8String]));
    NSError* error = nil;
    id<MTLComputePipelineState> p = [device newComputePipelineStateWithFunction:function error:&error];
    if (p == nil) {
        const char* msg = error ? [[error localizedDescription] UTF8String] : "unknown Metal pipeline error";
        throw std::runtime_error(std::string("Metal pipeline creation failed: ") + msg);
    }
    return p;
}

void dispatch_1d(id<MTLComputeCommandEncoder> encoder,
                 id<MTLComputePipelineState> p,
                 NSUInteger count) {
    const NSUInteger width = std::min<NSUInteger>(p.maxTotalThreadsPerThreadgroup, 256);
    [encoder setComputePipelineState:p];
    [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
}

} // namespace

void GdnMetalInputs::validate() const {
    if (T == 0 || num_k_heads == 0 || num_v_heads == 0 || head_dim == 0) {
        throw std::invalid_argument("GDN Metal dimensions must be non-zero");
    }
    if (!(head_dim == 16 || head_dim == 32 || head_dim == 64 || head_dim == 128)) {
        throw std::invalid_argument("GDN Metal head_dim must match QW3 supported dimensions (16/32/64/128)");
    }
    const size_t kd = static_cast<size_t>(T) * num_k_heads * head_dim;
    const size_t vd = static_cast<size_t>(T) * num_v_heads * head_dim;
    const size_t gd = static_cast<size_t>(T) * num_v_heads;
    const size_t sd = static_cast<size_t>(num_v_heads) * head_dim * head_dim;
    if (q.size() != kd || k.size() != kd || v.size() != vd ||
        alpha_raw.size() != gd || beta_raw.size() != gd ||
        dt_bias.size() != num_v_heads || ssm_a.size() != num_v_heads ||
        state.size() != sd) {
        throw std::invalid_argument("GDN Metal input shape mismatch");
    }
    validate_finite(q, "q");
    validate_finite(k, "k");
    validate_finite(v, "v");
    validate_finite(alpha_raw, "alpha_raw");
    validate_finite(beta_raw, "beta_raw");
    validate_finite(dt_bias, "dt_bias");
    validate_finite(ssm_a, "ssm_a");
    validate_finite(state, "state");
}

bool gdn_metal_available() {
    @autoreleasepool {
        return MTLCreateSystemDefaultDevice() != nil;
    }
}

GdnMetalResult gdn_metal_run(const GdnMetalInputs& inputs) {
    inputs.validate();
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) throw std::runtime_error("No Apple Metal device is available");

        NSString* shaderPath = [NSString stringWithUTF8String:QW3_SOURCE_DIR "/metal/gdn_reference.metal"];
        NSError* readError = nil;
        NSString* shader = [NSString stringWithContentsOfFile:shaderPath
                                                     encoding:NSUTF8StringEncoding
                                                        error:&readError];
        if (shader == nil) {
            const char* msg = readError ? [[readError localizedDescription] UTF8String] : "unknown read error";
            throw std::runtime_error(std::string("Unable to read Metal GDN shader: ") + msg);
        }

        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        options.fastMathEnabled = NO; // correctness profile; production tuning is separate.
        NSError* compileError = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:shader options:options error:&compileError];
        if (library == nil) {
            const char* msg = compileError ? [[compileError localizedDescription] UTF8String] : "unknown compiler error";
            throw std::runtime_error(std::string("Metal GDN shader compile failed: ") + msg);
        }

        auto prepare = pipeline(device, library, @"qw3_gdn_prepare");
        auto recurrent = pipeline(device, library, @"qw3_gdn_recurrent");
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (queue == nil) throw std::runtime_error("Unable to create Metal command queue");

        std::vector<float> alpha = inputs.alpha_raw;
        std::vector<float> beta = inputs.beta_raw;
        std::vector<float> state = inputs.state;
        std::vector<float> out(static_cast<size_t>(inputs.T) * inputs.num_v_heads * inputs.head_dim, 0.0f);

        id<MTLBuffer> qbuf = shared_buffer(device, inputs.q);
        id<MTLBuffer> kbuf = shared_buffer(device, inputs.k);
        id<MTLBuffer> vbuf = shared_buffer(device, inputs.v);
        id<MTLBuffer> abuf = shared_buffer(device, alpha);
        id<MTLBuffer> bbuf = shared_buffer(device, beta);
        id<MTLBuffer> dtbuf = shared_buffer(device, inputs.dt_bias);
        id<MTLBuffer> ssmbuf = shared_buffer(device, inputs.ssm_a);
        id<MTLBuffer> sbuf = shared_buffer(device, state);
        id<MTLBuffer> obuf = [device newBufferWithLength:out.size() * sizeof(float)
                                                options:MTLResourceStorageModeShared];
        if (!qbuf || !kbuf || !vbuf || !abuf || !bbuf || !dtbuf || !ssmbuf || !sbuf || !obuf) {
            throw std::runtime_error("Metal buffer allocation failed");
        }

        GdnDimsMetal dims{
            inputs.T,
            inputs.num_k_heads,
            inputs.num_v_heads,
            inputs.head_dim,
            inputs.num_k_heads * inputs.head_dim,
            inputs.num_v_heads * inputs.head_dim,
            inputs.num_v_heads,
            inputs.num_v_heads * inputs.head_dim,
        };
        const float scale = 1.0f / std::sqrt(static_cast<float>(inputs.head_dim));

        id<MTLCommandBuffer> command = [queue commandBuffer];
        if (command == nil) throw std::runtime_error("Unable to create Metal command buffer");

        id<MTLComputeCommandEncoder> pencoder = [command computeCommandEncoder];
        [pencoder setBuffer:abuf offset:0 atIndex:0];
        [pencoder setBuffer:bbuf offset:0 atIndex:1];
        [pencoder setBuffer:dtbuf offset:0 atIndex:2];
        [pencoder setBuffer:ssmbuf offset:0 atIndex:3];
        [pencoder setBytes:&dims length:sizeof(dims) atIndex:4];
        dispatch_1d(pencoder, prepare, static_cast<NSUInteger>(inputs.T) * inputs.num_v_heads);
        [pencoder endEncoding];

        id<MTLComputeCommandEncoder> rencoder = [command computeCommandEncoder];
        [rencoder setBuffer:qbuf offset:0 atIndex:0];
        [rencoder setBuffer:kbuf offset:0 atIndex:1];
        [rencoder setBuffer:vbuf offset:0 atIndex:2];
        [rencoder setBuffer:abuf offset:0 atIndex:3];
        [rencoder setBuffer:bbuf offset:0 atIndex:4];
        [rencoder setBuffer:sbuf offset:0 atIndex:5];
        [rencoder setBuffer:obuf offset:0 atIndex:6];
        [rencoder setBytes:&dims length:sizeof(dims) atIndex:7];
        [rencoder setBytes:&scale length:sizeof(scale) atIndex:8];
        dispatch_1d(rencoder, recurrent, static_cast<NSUInteger>(inputs.num_v_heads) * inputs.head_dim);
        [rencoder endEncoding];

        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) {
            const char* msg = command.error ? [[command.error localizedDescription] UTF8String] : "unknown command error";
            throw std::runtime_error(std::string("Metal GDN execution failed: ") + msg);
        }

        GdnMetalResult result;
        result.state.resize(state.size());
        result.out.resize(out.size());
        result.decay.resize(alpha.size());
        result.beta.resize(beta.size());
        std::memcpy(result.state.data(), sbuf.contents, result.state.size() * sizeof(float));
        std::memcpy(result.out.data(), obuf.contents, result.out.size() * sizeof(float));
        std::memcpy(result.decay.data(), abuf.contents, result.decay.size() * sizeof(float));
        std::memcpy(result.beta.data(), bbuf.contents, result.beta.size() * sizeof(float));
        result.device_name = [[device name] UTF8String];
        return result;
    }
}


void GdnMetalPreparedInputs::validate() const {
    if (T == 0 || num_k_heads == 0 || num_v_heads == 0 || head_dim == 0 ||
        num_v_heads % num_k_heads != 0) {
        throw std::invalid_argument("GDN Metal prepared dimensions are invalid");
    }
    if (!(head_dim == 16 || head_dim == 32 || head_dim == 64 || head_dim == 128)) {
        throw std::invalid_argument("GDN Metal prepared head_dim must be 16/32/64/128");
    }
    const size_t kd = static_cast<size_t>(T) * num_k_heads * head_dim;
    const size_t vd = static_cast<size_t>(T) * num_v_heads * head_dim;
    const size_t gd = static_cast<size_t>(T) * num_v_heads;
    if (k.size() != kd || v.size() != vd || decay.size() != gd || beta.size() != gd) {
        throw std::invalid_argument("GDN Metal prepared input shape mismatch");
    }
    validate_finite(k, "k");
    validate_finite(v, "v");
    validate_finite(decay, "decay");
    validate_finite(beta, "beta");
    for (float g : decay) if (g < 0.0f || g > 1.0001f) {
        throw std::invalid_argument("GDN prepared decay out of range");
    }
    for (float b : beta) if (b < 0.0f || b > 1.0f) {
        throw std::invalid_argument("GDN prepared beta out of range");
    }
}

GdnMetalAffineSummary gdn_metal_summarize(const GdnMetalPreparedInputs& inputs) {
    inputs.validate();
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) throw std::runtime_error("No Apple Metal device is available");

        NSString* shaderPath = [NSString stringWithUTF8String:QW3_SOURCE_DIR "/metal/gdn_reference.metal"];
        NSError* readError = nil;
        NSString* shader = [NSString stringWithContentsOfFile:shaderPath
                                                     encoding:NSUTF8StringEncoding
                                                        error:&readError];
        if (shader == nil) {
            const char* msg = readError ? [[readError localizedDescription] UTF8String] : "unknown read error";
            throw std::runtime_error(std::string("Unable to read Metal GDN shader: ") + msg);
        }
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        options.fastMathEnabled = NO;
        NSError* compileError = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:shader options:options error:&compileError];
        if (library == nil) {
            const char* msg = compileError ? [[compileError localizedDescription] UTF8String] : "unknown compiler error";
            throw std::runtime_error(std::string("Metal GDN shader compile failed: ") + msg);
        }
        auto summarize = pipeline(device, library, @"qw3_gdn_summarize");
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (queue == nil) throw std::runtime_error("Unable to create Metal command queue");

        const size_t per_head = static_cast<size_t>(inputs.head_dim) * inputs.head_dim;
        const size_t total = static_cast<size_t>(inputs.num_v_heads) * per_head;
        std::vector<float> T(total, 0.0f);
        std::vector<float> Z(total, 0.0f);
        for (uint32_t h = 0; h < inputs.num_v_heads; ++h) {
            for (uint32_t d = 0; d < inputs.head_dim; ++d) {
                T[static_cast<size_t>(h) * per_head + static_cast<size_t>(d) * inputs.head_dim + d] = 1.0f;
            }
        }

        id<MTLBuffer> kbuf = shared_buffer(device, inputs.k);
        id<MTLBuffer> vbuf = shared_buffer(device, inputs.v);
        id<MTLBuffer> gbuf = shared_buffer(device, inputs.decay);
        id<MTLBuffer> bbuf = shared_buffer(device, inputs.beta);
        id<MTLBuffer> tbuf = shared_buffer(device, T);
        id<MTLBuffer> zbuf = shared_buffer(device, Z);
        if (!kbuf || !vbuf || !gbuf || !bbuf || !tbuf || !zbuf) {
            throw std::runtime_error("Metal GDN summary buffer allocation failed");
        }

        GdnDimsMetal dims{
            inputs.T, inputs.num_k_heads, inputs.num_v_heads, inputs.head_dim,
            inputs.num_k_heads * inputs.head_dim,
            inputs.num_v_heads * inputs.head_dim,
            inputs.num_v_heads,
            inputs.num_v_heads * inputs.head_dim,
        };
        id<MTLCommandBuffer> command = [queue commandBuffer];
        if (command == nil) throw std::runtime_error("Unable to create Metal command buffer");
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setBuffer:kbuf offset:0 atIndex:0];
        [encoder setBuffer:vbuf offset:0 atIndex:1];
        [encoder setBuffer:gbuf offset:0 atIndex:2];
        [encoder setBuffer:bbuf offset:0 atIndex:3];
        [encoder setBuffer:tbuf offset:0 atIndex:4];
        [encoder setBuffer:zbuf offset:0 atIndex:5];
        [encoder setBytes:&dims length:sizeof(dims) atIndex:6];
        dispatch_1d(encoder, summarize, inputs.num_v_heads);
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) {
            const char* msg = command.error ? [[command.error localizedDescription] UTF8String] : "unknown command error";
            throw std::runtime_error(std::string("Metal GDN summary execution failed: ") + msg);
        }

        GdnMetalAffineSummary out;
        out.num_v_heads = inputs.num_v_heads;
        out.head_dim = inputs.head_dim;
        out.T.resize(total);
        out.Z.resize(total);
        std::memcpy(out.T.data(), tbuf.contents, total * sizeof(float));
        std::memcpy(out.Z.data(), zbuf.contents, total * sizeof(float));
        out.device_name = [[device name] UTF8String];
        return out;
    }
}

} // namespace qw3

namespace qw3 {

void GdnMetalSummaryChainInputs::validate() const {
    if (block_count == 0 || num_v_heads == 0 || head_dim == 0) {
        throw std::invalid_argument("GDN Metal summary-chain dimensions must be non-zero");
    }
    if (!(head_dim == 16 || head_dim == 32 || head_dim == 64 || head_dim == 128)) {
        throw std::invalid_argument("GDN Metal summary-chain head_dim must be 16/32/64/128");
    }
    const uint64_t per = static_cast<uint64_t>(num_v_heads) * head_dim * head_dim;
    const uint64_t all = static_cast<uint64_t>(block_count) * per;
    if (per > static_cast<uint64_t>(SIZE_MAX) || all > static_cast<uint64_t>(SIZE_MAX) ||
        T.size() != static_cast<size_t>(all) || Z.size() != static_cast<size_t>(all) ||
        state.size() != static_cast<size_t>(per)) {
        throw std::invalid_argument("GDN Metal summary-chain shape mismatch");
    }
    validate_finite(T, "summary_t");
    validate_finite(Z, "summary_z");
    validate_finite(state, "summary_state");
}

GdnMetalSummaryChainResult gdn_metal_apply_summary_chain(
    const GdnMetalSummaryChainInputs& inputs) {
    inputs.validate();
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) throw std::runtime_error("No Apple Metal device is available");

        NSString* shaderPath = [NSString stringWithUTF8String:QW3_SOURCE_DIR "/metal/gdn_reference.metal"];
        NSError* readError = nil;
        NSString* shader = [NSString stringWithContentsOfFile:shaderPath
                                                     encoding:NSUTF8StringEncoding
                                                        error:&readError];
        if (shader == nil) {
            const char* msg = readError ? [[readError localizedDescription] UTF8String] : "unknown read error";
            throw std::runtime_error(std::string("Unable to read Metal GDN shader: ") + msg);
        }
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        options.fastMathEnabled = NO;
        NSError* compileError = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:shader options:options error:&compileError];
        if (library == nil) {
            const char* msg = compileError ? [[compileError localizedDescription] UTF8String] : "unknown compiler error";
            throw std::runtime_error(std::string("Metal GDN shader compile failed: ") + msg);
        }
        auto apply = pipeline(device, library, @"qw3_gdn_apply_summary_chain");
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (queue == nil) throw std::runtime_error("Unable to create Metal command queue");

        std::vector<float> state = inputs.state;
        id<MTLBuffer> tbuf = shared_buffer(device, inputs.T);
        id<MTLBuffer> zbuf = shared_buffer(device, inputs.Z);
        id<MTLBuffer> sbuf = shared_buffer(device, state);
        if (!tbuf || !zbuf || !sbuf) throw std::runtime_error("Metal summary-chain buffer allocation failed");

        struct SummaryDims { uint32_t block_count, num_v_heads, head_dim; } dims{
            inputs.block_count, inputs.num_v_heads, inputs.head_dim};
        id<MTLCommandBuffer> command = [queue commandBuffer];
        if (command == nil) throw std::runtime_error("Unable to create Metal command buffer");
        id<MTLComputeCommandEncoder> enc = [command computeCommandEncoder];
        [enc setBuffer:tbuf offset:0 atIndex:0];
        [enc setBuffer:zbuf offset:0 atIndex:1];
        [enc setBuffer:sbuf offset:0 atIndex:2];
        [enc setBytes:&dims length:sizeof(dims) atIndex:3];
        dispatch_1d(enc, apply, static_cast<NSUInteger>(inputs.num_v_heads) * inputs.head_dim);
        [enc endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) {
            const char* msg = command.error ? [[command.error localizedDescription] UTF8String] : "unknown command error";
            throw std::runtime_error(std::string("Metal GDN summary-chain execution failed: ") + msg);
        }
        GdnMetalSummaryChainResult out;
        out.state.resize(state.size());
        std::memcpy(out.state.data(), sbuf.contents, out.state.size() * sizeof(float));
        out.device_name = [[device name] UTF8String];
        return out;
    }
}

} // namespace qw3
