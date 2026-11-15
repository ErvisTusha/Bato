// Direct Apple Metal compute implementation for M-series GPUs.
// Uses Metal Objective-C APIs and MSL kernels, not wgpu.
// Provenance: original Bato filters using Apple's public Metal API.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "../include/bato_gpu_backend.h"
#include <cstring>
#include <cstdint>

namespace {
constexpr uint32_t kTile = 4096;
struct Params {
    uint32_t length;
    uint32_t kind;
    uint32_t inverse;
    uint32_t distance;
    uint32_t tiles;
};

static const char *kShader = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct Params {
    uint length;
    uint kind;
    uint inverse;
    uint distance;
    uint tiles;
};
constant uint TILE_BYTES = 4096;

kernel void direct_filter(
    device const uchar *source [[buffer(0)]],
    device uchar *target [[buffer(1)]],
    constant Params &p [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= p.length) return;
    if (p.kind >= 5) {
        const uint words = p.length / p.distance;
        const uint body = words * p.distance;
        const uint src_index = i >= body ? i :
            (p.inverse ? ((i % p.distance) * words + i / p.distance)
                       : ((i % words) * p.distance + i / words));
        target[i] = source[src_index];
    } else {
        target[i] = uchar(uint(source[i]) -
            (i >= p.distance ? uint(source[i - p.distance]) : 0u));
    }
}

kernel void delta_local(
    device const uchar *source [[buffer(0)]],
    device uchar *target [[buffer(1)]],
    device uchar *totals [[buffer(2)]],
    constant Params &p [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    const uint tile = id / p.distance;
    const uint lane = id % p.distance;
    const uint begin = tile * TILE_BYTES;
    if (begin >= p.length) return;
    const uint end = min(begin + TILE_BYTES, p.length);
    uint accumulator = 0;
    for (uint i = begin + lane; i < end; i += p.distance) {
        accumulator += uint(source[i]);
        target[i] = uchar(accumulator);
    }
    totals[id] = uchar(accumulator);
}

kernel void delta_carries(
    device uchar *totals [[buffer(2)]],
    constant Params &p [[buffer(3)]],
    uint lane [[thread_position_in_grid]]
) {
    if (lane >= p.distance) return;
    uint accumulator = 0;
    for (uint tile = 0; tile < p.tiles; ++tile) {
        const uint i = tile * p.distance + lane;
        const uint amount = uint(totals[i]);
        totals[i] = uchar(accumulator);
        accumulator += amount;
    }
}

kernel void delta_finish(
    device uchar *target [[buffer(1)]],
    device const uchar *totals [[buffer(2)]],
    constant Params &p [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= p.length) return;
    target[i] = uchar(uint(target[i]) +
        uint(totals[(i / TILE_BYTES) * p.distance + i % p.distance]));
}
)METAL";

uint32_t distance(uint32_t transform) {
    switch (transform) {
        case BATO_FILTER_DELTA1: return 1;
        case BATO_FILTER_DELTA2:
        case BATO_FILTER_SHUFFLE2: return 2;
        case BATO_FILTER_DELTA4:
        case BATO_FILTER_SHUFFLE4: return 4;
        default: return 0;
    }
}

void run_pass(
    id<MTLCommandBuffer> command,
    id<MTLComputePipelineState> pipeline,
    id<MTLBuffer> input, id<MTLBuffer> output,
    id<MTLBuffer> totals, id<MTLBuffer> parameters,
    NSUInteger threads
) {
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    if (input) [encoder setBuffer:input offset:0 atIndex:0];
    if (output) [encoder setBuffer:output offset:0 atIndex:1];
    if (totals) [encoder setBuffer:totals offset:0 atIndex:2];
    [encoder setBuffer:parameters offset:0 atIndex:3];
    const NSUInteger group = MIN(256u, [pipeline maxTotalThreadsPerThreadgroup]);
    [encoder dispatchThreads:MTLSizeMake(threads, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(group, 1, 1)];
    [encoder endEncoding];
}
}

BATO_GPU_API uint32_t bato_gpu_abi_version(void) {
    return BATO_GPU_ABI_VERSION;
}

BATO_GPU_API int32_t bato_gpu_device_count(void) {
#if defined(__arm64__) || defined(__aarch64__)
    @autoreleasepool {
        return (int32_t)[MTLCopyAllDevices() count];
    }
#else
    return 0;
#endif
}

BATO_GPU_API int32_t bato_gpu_filter(
    uint32_t ordinal, const uint8_t *input, uint8_t *output,
    size_t len, uint32_t transform, uint32_t inverse
) {
#if !defined(__arm64__) && !defined(__aarch64__)
    (void)ordinal; (void)input; (void)output;
    (void)len; (void)transform; (void)inverse;
    return -10;
#else
    if (!input || !output || len == 0 || len > 64u * 1024u * 1024u ||
        transform < BATO_FILTER_DELTA1 || transform > BATO_FILTER_SHUFFLE4 ||
        inverse > 1u) return -1;
    @autoreleasepool {
        @try {
            NSArray<id<MTLDevice>> *devices = MTLCopyAllDevices();
            if (ordinal >= [devices count]) return -2;
            id<MTLDevice> gpu = devices[ordinal];
            NSError *error = nil;
            NSString *source = [NSString stringWithUTF8String:kShader];
            id<MTLLibrary> library = [gpu newLibraryWithSource:source
                                                       options:nil error:&error];
            if (!library) return -3;
            const NSArray<NSString *> *names =
                @[@"direct_filter", @"delta_local", @"delta_carries", @"delta_finish"];
            NSMutableArray<id<MTLComputePipelineState>> *pipelines =
                [NSMutableArray arrayWithCapacity:4];
            for (NSString *name in names) {
                id<MTLFunction> function = [library newFunctionWithName:name];
                if (!function) return -4;
                id<MTLComputePipelineState> pipeline =
                    [gpu newComputePipelineStateWithFunction:function error:&error];
                if (!pipeline) return -5;
                [pipelines addObject:pipeline];
            }
            id<MTLBuffer> src = [gpu newBufferWithBytes:input length:len
                                                options:MTLResourceStorageModeShared];
            id<MTLBuffer> dst = [gpu newBufferWithLength:len
                                                options:MTLResourceStorageModeShared];
            if (!src || !dst) return -6;
            Params p = {
                (uint32_t)len, transform, inverse, distance(transform),
                ((uint32_t)len + kTile - 1) / kTile
            };
            id<MTLBuffer> params = [gpu newBufferWithBytes:&p length:sizeof(p)
                                                   options:MTLResourceStorageModeShared];
            id<MTLBuffer> carry = nil;
            if (inverse && transform <= BATO_FILTER_DELTA4) {
                carry = [gpu newBufferWithLength:(NSUInteger)p.tiles * p.distance
                                        options:MTLResourceStorageModeShared];
                if (!carry) return -7;
            }
            id<MTLCommandQueue> queue = [gpu newCommandQueue];
            id<MTLCommandBuffer> command = [queue commandBuffer];
            if (!queue || !command) return -8;
            if (carry) {
                run_pass(command, pipelines[1], src, dst, carry, params,
                         (NSUInteger)p.tiles * p.distance);
                run_pass(command, pipelines[2], nil, nil, carry, params, p.distance);
                run_pass(command, pipelines[3], nil, dst, carry, params, p.length);
            } else {
                run_pass(command, pipelines[0], src, dst, nil, params, p.length);
            }
            [command commit];
            [command waitUntilCompleted];
            if ([command status] != MTLCommandBufferStatusCompleted) return -9;
            std::memcpy(output, [dst contents], len);
            return 0;
        } @catch (NSException *exception) {
            (void)exception;
            return -11;
        }
    }
#endif
}
