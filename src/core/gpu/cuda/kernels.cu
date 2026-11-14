// Bato experimental CUDA kernels. Provenance: original. They compute the same
// integers as `matvec::reference` and `streams::CpuStreams`.
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
