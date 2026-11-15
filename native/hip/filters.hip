// AMD HIP runtime implementation of Bato's five reversible byte filters.
// Uses ROCm/HIP kernels, not wgpu. Provenance: original implementation based
// on the public HIP runtime programming API and Bato's byte-filter semantics.
#include "../include/bato_gpu_backend.h"
#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {
constexpr uint32_t kBlock = 256;
constexpr uint32_t kTile = 4096;

uint32_t distance(uint32_t transform) {
    switch (transform) {
        case BATO_FILTER_DELTA1: return 1;
        case BATO_FILTER_DELTA2: case BATO_FILTER_SHUFFLE2: return 2;
        case BATO_FILTER_DELTA4: case BATO_FILTER_SHUFFLE4: return 4;
        default: return 0;
    }
}

__global__ void filter_direct(
    const uint8_t *source, uint8_t *target, uint32_t length,
    uint32_t transform, uint32_t inverse
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= length) return;
    const uint32_t d = distance(transform);
    if (transform >= BATO_FILTER_SHUFFLE2) {
        const uint32_t words = length / d;
        const uint32_t body = words * d;
        if (i >= body) { target[i] = source[i]; return; }
        const uint32_t src = inverse
            ? ((i % d) * words + (i / d))
            : ((i % words) * d + (i / words));
        target[i] = source[src];
    } else {
        target[i] = static_cast<uint8_t>(
            source[i] - (i >= d ? source[i - d] : 0)
        );
    }
}

// One independent residue chain per tile: 4096 is divisible by 1, 2, 4.
__global__ void delta_local(
    const uint8_t *source, uint8_t *target, uint8_t *totals,
    uint32_t length, uint32_t d
) {
    const uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t tile = id / d;
    const uint32_t lane = id % d;
    const uint32_t begin = tile * kTile;
    if (begin >= length) return;
    const uint32_t end = begin + kTile < length ? begin + kTile : length;
    uint32_t accumulator = 0;
    for (uint32_t i = begin + lane; i < end; i += d) {
        accumulator += source[i];
        target[i] = static_cast<uint8_t>(accumulator);
    }
    totals[id] = static_cast<uint8_t>(accumulator);
}

__global__ void delta_carries(uint8_t *totals, uint32_t tiles, uint32_t d) {
    const uint32_t lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= d) return;
    uint32_t accumulator = 0;
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        const uint32_t i = tile * d + lane;
        const uint32_t total = totals[i];
        totals[i] = static_cast<uint8_t>(accumulator);
        accumulator += total;
    }
}

__global__ void delta_finish(
    uint8_t *target, const uint8_t *totals, uint32_t length, uint32_t d
) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= length) return;
    const uint32_t tile = i / kTile;
    const uint32_t lane = i % d;
    target[i] = static_cast<uint8_t>(
        static_cast<uint32_t>(target[i]) + totals[tile * d + lane]
    );
}

struct DeviceBuffer {
    void *data = nullptr;
    ~DeviceBuffer() { if (data) (void)hipFree(data); }
    hipError_t allocate(size_t bytes) { return hipMalloc(&data, bytes); }
};

bool supported(uint32_t kind) {
    return kind >= BATO_FILTER_DELTA1 && kind <= BATO_FILTER_SHUFFLE4;
}

bool check(hipError_t result) { return result == hipSuccess; }
} // namespace

BATO_GPU_API uint32_t bato_gpu_abi_version(void) {
    return BATO_GPU_ABI_VERSION;
}

BATO_GPU_API int32_t bato_gpu_device_count(void) {
    int count = 0;
    if (!check(hipGetDeviceCount(&count))) return 0;
    return count;
}

BATO_GPU_API int32_t bato_gpu_filter(
    uint32_t device, const uint8_t *input, uint8_t *output,
    size_t len, uint32_t transform, uint32_t inverse
) {
    if (!input || !output || len == 0 || len > 64u * 1024u * 1024u ||
        !supported(transform) || inverse > 1u) return -1;
    int count = bato_gpu_device_count();
    if (device >= static_cast<uint32_t>(count)) return -2;
    if (!check(hipSetDevice(static_cast<int>(device)))) return -3;

    DeviceBuffer source, target, carries;
    if (!check(source.allocate(len)) || !check(target.allocate(len))) return -4;
    if (!check(hipMemcpy(source.data, input, len, hipMemcpyHostToDevice))) return -5;

    const uint32_t length = static_cast<uint32_t>(len);
    const uint32_t d = distance(transform);
    const uint32_t blocks = (length + kBlock - 1) / kBlock;
    if (inverse && transform <= BATO_FILTER_DELTA4) {
        const uint32_t tiles = (length + kTile - 1) / kTile;
        if (!check(carries.allocate(static_cast<size_t>(tiles) * d))) return -6;
        const uint32_t chains = tiles * d;
        hipLaunchKernelGGL(delta_local, dim3((chains + kBlock - 1) / kBlock),
                           dim3(kBlock), 0, 0,
                           static_cast<const uint8_t *>(source.data),
                           static_cast<uint8_t *>(target.data),
                           static_cast<uint8_t *>(carries.data), length, d);
        hipLaunchKernelGGL(delta_carries, dim3(1), dim3(kBlock), 0, 0,
                           static_cast<uint8_t *>(carries.data), tiles, d);
        hipLaunchKernelGGL(delta_finish, dim3(blocks), dim3(kBlock), 0, 0,
                           static_cast<uint8_t *>(target.data),
                           static_cast<const uint8_t *>(carries.data), length, d);
    } else {
        hipLaunchKernelGGL(filter_direct, dim3(blocks), dim3(kBlock), 0, 0,
                           static_cast<const uint8_t *>(source.data),
                           static_cast<uint8_t *>(target.data),
                           length, transform, inverse);
    }
    if (!check(hipGetLastError()) || !check(hipDeviceSynchronize())) return -7;
    if (!check(hipMemcpy(output, target.data, len, hipMemcpyDeviceToHost))) return -8;
    return 0;
}
