// Bato experimental CUDA kernels. Provenance: original. They compute the same
// integers as the Rust matrix-vector, stream and byte-filter implementations.
//
// `kernels.ptx` is generated from this file and loaded by the NVIDIA driver;
// neither the CUDA Toolkit nor its headers are needed. Regenerate it with:
//
//   cd src/core/gpu/cuda
//   clang -x cuda --cuda-device-only -nocudainc -nocudalib --cuda-gpu-arch=sm_75 \
//     -O3 -S -o body.ptx kernels.cu
//   { echo "// source sha256: $(sha256sum < kernels.cu | cut -d' ' -f1)"; cat body.ptx; } \
//     > kernels.ptx && rm body.ptx
//
// sm_75 is the oldest architecture the measured device (GTX 1650) supports.

typedef unsigned int u32;
typedef unsigned long long u64;
typedef long long i64;

#define OUTPUT_SHIFT 10
#define OUTPUT_LIMIT 2047
#define CELL_ONE 65535u
#define RATE_SHIFT 4

// Index of this thread over the whole launch.
__attribute__((device)) static inline u32 thread_index() {
    return __nvvm_read_ptx_sreg_ctaid_x() * __nvvm_read_ptx_sreg_ntid_x() +
           __nvvm_read_ptx_sreg_tid_x();
}

// Does nothing: launching it measures the cost of a call alone.
extern "C" __attribute__((global)) void empty() {}

// One output row per thread: clamp(sum(weight * input) >> OUTPUT_SHIFT).
extern "C" __attribute__((global)) void matvec(const short *weights, const short *inputs,
                                               int *outputs, u32 rows, u32 cols) {
    u32 row = thread_index();
    if (row >= rows) {
        return;
    }
    const short *line = weights + (u64)row * cols;
    i64 sum = 0;
    for (u32 col = 0; col < cols; col++) {
        sum += (i64)line[col] * inputs[col];
    }
    sum >>= OUTPUT_SHIFT;
    if (sum > OUTPUT_LIMIT) {
        sum = OUTPUT_LIMIT;
    }
    if (sum < -OUTPUT_LIMIT) {
        sum = -OUTPUT_LIMIT;
    }
    outputs[row] = (int)sum;
}

__attribute__((device)) static inline u32 mix(u32 x) {
    x ^= x >> 16;
    x *= 0x9e3779b1u;
    x ^= x >> 13;
    x *= 0x85ebca6bu;
    x ^= x >> 16;
    return x;
}

// One serial stream per thread: `steps` dependent lookups in its own table.
extern "C" __attribute__((global)) void streams(u32 *table, u32 *states, u32 count,
                                                u32 cells_log2, u32 steps) {
    u32 stream = thread_index();
    if (stream >= count) {
        return;
    }
    u32 *cells = table + ((u64)stream << cells_log2);
    u32 mask = (1u << cells_log2) - 1u;
    u32 state = states[stream];
    for (u32 step = 0; step < steps; step++) {
        u32 slot = mix(state) & mask;
        u32 cell = cells[slot];
        u32 goal = (state & 1u) * CELL_ONE;
        u32 next = (u32)((int)cell + (((int)goal - (int)cell) >> RATE_SHIFT));
        cells[slot] = next;
        state = mix(state + next + 1u);
    }
    states[stream] = state;
}

// Exact delta and byte-plane filters (the format operations in cm/filters.rs).
// Each invocation writes a disjoint byte range.
typedef unsigned char u8;
#define FILTER_THREADS 64u
#define DELTA_THREAD_BYTES 64u
#define DELTA_TILE_BYTES (FILTER_THREADS * DELTA_THREAD_BYTES)

extern "C" __attribute__((global)) void filter_bytes(const u8 *input, u8 *output,
                                                     u32 size, u32 transform) {
    u32 i = thread_index();
    if (i >= size) {
        return;
    }
    // Transform identifiers: 2..4 are forward delta, 5..6 forward shuffle,
    // and 7..8 inverse shuffle (the latter are internal to this kernel).
    if (transform <= 4) {
        u32 distance = 1u << (transform - 2);
        output[i] = input[i] - (i < distance ? 0 : input[i - distance]);
        return;
    }
    u32 width = 2u << ((transform - 5) & 1);
    u32 words = size / width;
    u32 from = i;
    if (i < words * width) {
        from = transform <= 6 ? (i % words) * width + i / words
                              : (i % width) * words + i / width;
    }
    output[i] = input[from];
}

// Add independent byte lanes modulo 256. Separating alternating lanes
// prevents an overflow in one lane from changing the next.
__attribute__((device)) static inline u32 add_lanes(u32 a, u32 b) {
    const u32 mask = 0x00ff00ffu;
    return (((a & mask) + (b & mask)) & mask) |
           ((((a >> 8) & mask) + ((b >> 8) & mask)) & mask) << 8;
}

// Invert delta within fixed tiles using a parallel prefix sum. A thread
// first obtains its local sum, then scans the previous threads' totals.
// Each tile also returns the sum needed by the next tile.
extern "C" __attribute__((global)) void filter_delta_local(const u8 *input, u8 *output,
                                                           u32 *tiles, u32 size, u32 distance) {
    __attribute__((shared)) u32 totals[FILTER_THREADS];
    const u32 lane = __nvvm_read_ptx_sreg_tid_x();
    const u32 tile = __nvvm_read_ptx_sreg_ctaid_x();
    const u32 start = (tile * FILTER_THREADS + lane) * DELTA_THREAD_BYTES;
    const u32 end = start + DELTA_THREAD_BYTES < size ? start + DELTA_THREAD_BYTES : size;
    const u32 mask = distance - 1;
    u8 accum[4] = {};
    for (u32 i = start; i < end; i++) {
        accum[i & mask] += input[i];
    }
    totals[lane] = (u32)accum[0] | (u32)accum[1] << 8 | (u32)accum[2] << 16 | (u32)accum[3] << 24;
    asm volatile("bar.sync 0;" ::: "memory");
    for (u32 offset = 1; offset < FILTER_THREADS; offset *= 2) {
        const u32 prior = lane >= offset ? totals[lane - offset] : 0;
        asm volatile("bar.sync 0;" ::: "memory");
        totals[lane] = add_lanes(totals[lane], prior);
        asm volatile("bar.sync 0;" ::: "memory");
    }
    if (lane == FILTER_THREADS - 1) {
        tiles[tile] = totals[lane];
    }
    const u32 carry = lane != 0 ? totals[lane - 1] : 0;
    for (u32 k = 0; k < 4; k++) {
        accum[k] = carry >> (8 * k);
    }
    for (u32 i = start; i < end; i++) {
        accum[i & mask] += input[i];
        output[i] = accum[i & mask];
    }
}

extern "C" __attribute__((global)) void filter_delta_carries(u32 *tiles, u32 count) {
    if (thread_index() != 0) {
        return;
    }
    u32 carry = 0;
    for (u32 i = 0; i < count; i++) {
        const u32 next = add_lanes(carry, tiles[i]);
        tiles[i] = carry;
        carry = next;
    }
}

extern "C" __attribute__((global)) void filter_delta_finish(u8 *output, const u32 *tiles,
                                                            u32 size, u32 distance) {
    const u32 i = thread_index();
    if (i < size) {
        output[i] += (u8)(tiles[i / DELTA_TILE_BYTES] >> (8 * (i & (distance - 1))));
    }
}
