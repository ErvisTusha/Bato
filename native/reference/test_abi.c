/* Test-only CPU ABI fixture for validating the Rust native plugin boundary.
 * Not a GPU backend and never loaded by Bato outside explicit test setup.
 * Provenance: original Bato filter semantics.
 */
#include "../include/bato_gpu_backend.h"

BATO_GPU_API uint32_t bato_gpu_abi_version(void) {
    return BATO_GPU_ABI_VERSION;
}

BATO_GPU_API int32_t bato_gpu_device_count(void) {
    return 1;
}

BATO_GPU_API int32_t bato_gpu_filter(
    uint32_t device, const uint8_t *input, uint8_t *output,
    size_t len, uint32_t transform, uint32_t inverse
) {
    if (device != 0 || !input || !output || len == 0 ||
        transform < BATO_FILTER_DELTA1 || transform > BATO_FILTER_SHUFFLE4 ||
        inverse > 1) return -1;
    const size_t width = transform == BATO_FILTER_DELTA1 ? 1 :
        (transform == BATO_FILTER_DELTA2 || transform == BATO_FILTER_SHUFFLE2 ? 2 : 4);
    if (transform <= BATO_FILTER_DELTA4) {
        for (size_t i = 0; i < len; ++i) {
            const uint8_t previous = i >= width
                ? (inverse ? output[i - width] : input[i - width]) : 0;
            output[i] = (uint8_t)(inverse
                ? (unsigned)input[i] + previous
                : (unsigned)input[i] - previous);
        }
    } else {
        const size_t words = len / width;
        const size_t body = words * width;
        for (size_t i = 0; i < len; ++i) {
            const size_t src = i >= body ? i : (
                inverse
                ? (i % width) * words + (i / width)
                : (i % words) * width + (i / words)
            );
            output[i] = input[src];
        }
    }
    return 0;
}
