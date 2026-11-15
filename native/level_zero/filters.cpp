// Native Intel Level Zero execution through Intel's oneAPI SYCL backend.
// The selected SYCL devices must report ext_oneapi_level_zero, never OpenCL.
// Provenance: original Bato filters using public oneAPI Level Zero backend APIs.
#include "../include/bato_gpu_backend.h"
#include <sycl/sycl.hpp>
#include <cstdint>
#include <vector>

namespace {
constexpr uint32_t kTile = 4096;

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

std::vector<sycl::device> devices() {
    std::vector<sycl::device> found;
    for (const auto &platform : sycl::platform::get_platforms()) {
        if (platform.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
        for (const auto &device : platform.get_devices(sycl::info::device_type::gpu)) {
            found.push_back(device);
        }
    }
    return found;
}

struct Buffers {
    sycl::queue &queue;
    uint8_t *source = nullptr;
    uint8_t *target = nullptr;
    uint8_t *totals = nullptr;
    ~Buffers() {
        if (source) sycl::free(source, queue);
        if (target) sycl::free(target, queue);
        if (totals) sycl::free(totals, queue);
    }
};
}

BATO_GPU_API uint32_t bato_gpu_abi_version(void) {
    return BATO_GPU_ABI_VERSION;
}

BATO_GPU_API int32_t bato_gpu_device_count(void) {
    try {
        return static_cast<int32_t>(devices().size());
    } catch (...) {
        return 0;
    }
}

BATO_GPU_API int32_t bato_gpu_filter(
    uint32_t device, const uint8_t *input, uint8_t *output,
    size_t len, uint32_t transform, uint32_t inverse
) {
    if (!input || !output || len == 0 || len > 64u * 1024u * 1024u ||
        transform < BATO_FILTER_DELTA1 || transform > BATO_FILTER_SHUFFLE4 ||
        inverse > 1u) return -1;
    try {
        const auto list = devices();
        if (device >= list.size()) return -2;
        sycl::queue queue(list[device]);
        Buffers buffers{queue};
        buffers.source = sycl::malloc_device<uint8_t>(len, queue);
        buffers.target = sycl::malloc_device<uint8_t>(len, queue);
        if (!buffers.source || !buffers.target) return -3;
        queue.memcpy(buffers.source, input, len).wait();

        const uint32_t length = static_cast<uint32_t>(len);
        const uint32_t d = distance(transform);
        const uint8_t *src = buffers.source;
        uint8_t *dst = buffers.target;
        if (inverse && transform <= BATO_FILTER_DELTA4) {
            const uint32_t tiles = (length + kTile - 1) / kTile;
            buffers.totals = sycl::malloc_device<uint8_t>(
                static_cast<size_t>(tiles) * d, queue
            );
            if (!buffers.totals) return -4;
            uint8_t *totals = buffers.totals;
            queue.parallel_for(sycl::range<1>(tiles * d), [=](sycl::id<1> id) {
                const uint32_t chain = static_cast<uint32_t>(id[0]);
                const uint32_t tile = chain / d;
                const uint32_t lane = chain % d;
                const uint32_t begin = tile * kTile;
                const uint32_t end = begin + kTile < length ? begin + kTile : length;
                uint32_t accumulator = 0;
                for (uint32_t i = begin + lane; i < end; i += d) {
                    accumulator += src[i];
                    dst[i] = static_cast<uint8_t>(accumulator);
                }
                totals[chain] = static_cast<uint8_t>(accumulator);
            }).wait();
            queue.parallel_for(sycl::range<1>(d), [=](sycl::id<1> id) {
                const uint32_t lane = static_cast<uint32_t>(id[0]);
                uint32_t accumulator = 0;
                for (uint32_t tile = 0; tile < tiles; ++tile) {
                    const uint32_t i = tile * d + lane;
                    const uint32_t total = totals[i];
                    totals[i] = static_cast<uint8_t>(accumulator);
                    accumulator += total;
                }
            }).wait();
            queue.parallel_for(sycl::range<1>(length), [=](sycl::id<1> id) {
                const uint32_t i = static_cast<uint32_t>(id[0]);
                const uint32_t lane = i % d;
                dst[i] = static_cast<uint8_t>(
                    static_cast<uint32_t>(dst[i]) + totals[(i / kTile) * d + lane]
                );
            }).wait();
        } else {
            queue.parallel_for(sycl::range<1>(length), [=](sycl::id<1> id) {
                const uint32_t i = static_cast<uint32_t>(id[0]);
                if (transform >= BATO_FILTER_SHUFFLE2) {
                    const uint32_t words = length / d;
                    const uint32_t body = words * d;
                    const uint32_t source_index = i >= body ? i : (
                        inverse
                        ? ((i % d) * words + i / d)
                        : ((i % words) * d + i / words)
                    );
                    dst[i] = src[source_index];
                } else {
                    dst[i] = static_cast<uint8_t>(
                        src[i] - (i >= d ? src[i - d] : 0)
                    );
                }
            }).wait();
        }
        queue.memcpy(output, dst, len).wait();
        return 0;
    } catch (...) {
        return -5;
    }
}
