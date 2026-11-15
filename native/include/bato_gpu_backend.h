/* Bato GPU accelerator plugin ABI v1.
 * The library is loaded dynamically and is never part of the archive format.
 * Provenance: original Bato integration and the public platform GPU APIs.
 */
#ifndef BATO_GPU_BACKEND_H
#define BATO_GPU_BACKEND_H

#include <stddef.h>
#include <stdint.h>

#if defined(__cplusplus)
#define BATO_GPU_EXPORT extern "C"
#else
#define BATO_GPU_EXPORT extern
#endif
#if defined(_WIN32)
#define BATO_GPU_VISIBLE __declspec(dllexport)
#else
#define BATO_GPU_VISIBLE __attribute__((visibility("default")))
#endif
#define BATO_GPU_API BATO_GPU_EXPORT BATO_GPU_VISIBLE

#define BATO_GPU_ABI_VERSION 1u
#define BATO_FILTER_DELTA1 2u
#define BATO_FILTER_DELTA2 3u
#define BATO_FILTER_DELTA4 4u
#define BATO_FILTER_SHUFFLE2 5u
#define BATO_FILTER_SHUFFLE4 6u

/* Native GPU API implementations must use exactly these exports.
 * device_count() returns 0 when no compatible hardware is present.
 * filter() is synchronous and must not retain the host pointers.
 * filter() returns 0 on success, nonzero on any error or unsupported filter.
 * A failure must not be treated as successful compression/decompression.
 * Inputs and outputs do not overlap. len is between 1 and 64 MiB.
 */
BATO_GPU_API uint32_t bato_gpu_abi_version(void);
BATO_GPU_API int32_t bato_gpu_device_count(void);
BATO_GPU_API int32_t bato_gpu_filter(
    uint32_t device, const uint8_t *input, uint8_t *output,
    size_t len, uint32_t transform, uint32_t inverse
);

#endif
