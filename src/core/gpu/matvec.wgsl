// Integer matrix-vector product, one output row per invocation. The 64-bit
// sum is kept in two 32-bit limbs because WGSL has no 64-bit integers; every
// operation is exact, so the result equals `matvec::reference` bit for bit.

const OUTPUT_SHIFT: u32 = 10u;
const OUTPUT_LIMIT: i32 = 2047;

struct Params {
    rows: u32,
    cols: u32,
}

@group(0) @binding(0) var<storage, read> weights: array<i32>;
@group(0) @binding(1) var<storage, read> inputs: array<i32>;
@group(0) @binding(2) var<storage, read_write> outputs: array<i32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = id.x;
    if (row >= params.rows) {
        return;
    }
    let base = row * params.cols;
    var low: u32 = 0u;
    var high: i32 = 0;
    for (var col: u32 = 0u; col < params.cols; col++) {
        // Both factors are 16-bit values, so the product fits 32 bits.
        let product = weights[base + col] * inputs[col];
        let next = low + bitcast<u32>(product);
        // Sign extension of the product plus the carry out of the low limb.
        high += (product >> 31u) + i32(next < low);
        low = next;
    }
    let top = high >> OUTPUT_SHIFT;
    let shifted = bitcast<i32>((bitcast<u32>(high) << (32u - OUTPUT_SHIFT)) | (low >> OUTPUT_SHIFT));
    var value: i32 = OUTPUT_LIMIT;
    if (high < 0) {
        value = -OUTPUT_LIMIT;
    }
    if ((top == 0 && shifted >= 0) || (top == -1 && shifted < 0)) {
        value = clamp(shifted, -OUTPUT_LIMIT, OUTPUT_LIMIT);
    }
    outputs[row] = value;
}
