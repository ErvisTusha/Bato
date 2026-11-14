// Independent serial streams, one per invocation. Every step reads a cell of
// the stream's own table at a hashed position, adapts it and derives the next
// position from the value read, like the lookups of a context model. All
// arithmetic is exact 32-bit, so the states equal `streams::CpuStreams`.

const CELL_ONE: u32 = 65535u;
const RATE_SHIFT: u32 = 4u;

struct Params {
    streams: u32,
    cells_log2: u32,
    steps: u32,
}

@group(0) @binding(0) var<storage, read_write> table: array<u32>;
@group(0) @binding(1) var<storage, read_write> states: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

fn mix(value: u32) -> u32 {
    var x = value;
    x ^= x >> 16u;
    x *= 0x9e3779b1u;
    x ^= x >> 13u;
    x *= 0x85ebca6bu;
    x ^= x >> 16u;
    return x;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let stream = id.x;
    if (stream >= params.streams) {
        return;
    }
    let base = stream << params.cells_log2;
    let mask = (1u << params.cells_log2) - 1u;
    var state = states[stream];
    for (var step: u32 = 0u; step < params.steps; step++) {
        let slot = base + (mix(state) & mask);
        let cell = table[slot];
        let goal = (state & 1u) * CELL_ONE;
        let next = u32(i32(cell) + ((i32(goal) - i32(cell)) >> RATE_SHIFT));
        table[slot] = next;
        state = mix(state + next + 1u);
    }
    states[stream] = state;
}
