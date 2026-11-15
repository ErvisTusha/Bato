// Original integer port of Bato's delta and byte-plane transforms.
// Each thread owns packed output words, including the final partial word.
struct Filter {
    size: u32,
    kind: u32,
    word_groups_x: u32,
    tile_groups_x: u32,
}
@group(0) @binding(0) var<storage, read> source: array<u32>;
@group(0) @binding(1) var<storage, read_write> result: array<u32>;
@group(0) @binding(2) var<uniform> parameters: Filter;
@group(0) @binding(3) var<storage, read_write> tiles: array<u32>;
var<workgroup> totals: array<u32, 64>;

fn byte_at(offset: u32) -> u32 {
    return (source[offset / 4u] >> (8u * (offset % 4u))) & 255u;
}

fn add_lanes(a: u32, b: u32) -> u32 {
    let mask = 0x00ff00ffu;
    return (((a & mask) + (b & mask)) & mask)
        | (((((a >> 8u) & mask) + ((b >> 8u) & mask)) & mask) << 8u);
}

fn words() -> u32 {
    return (parameters.size + 3u) / 4u;
}

fn tile_count() -> u32 {
    return (words() + 1023u) / 1024u;
}

@compute @workgroup_size(64)
fn direct(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let word = (group.y * parameters.word_groups_x + group.x) * 64u + lane;
    if word >= words() { return; }
    var packed = 0u;
    for (var byte = 0u; byte < 4u; byte++) {
        let i = word * 4u + byte;
        if i >= parameters.size { break; }
        var value = byte_at(i);
        if parameters.kind <= 4u {
            let distance = 1u << (parameters.kind - 2u);
            if i >= distance { value = value - byte_at(i - distance); }
        } else {
            let width = 2u << ((parameters.kind - 5u) & 1u);
            let count = parameters.size / width;
            if i < count * width {
                if parameters.kind <= 6u {
                    value = byte_at((i % count) * width + i / count);
                } else {
                    value = byte_at((i % width) * count + i / width);
                }
            }
        }
        packed |= (value & 255u) << (8u * byte);
    }
    result[word] = packed;
}

// Accumulate one input word's four bytes into their independent delta lanes.
fn residue_sum(value: u32, distance: u32) -> u32 {
    var sum = 0u;
    for (var byte = 0u; byte < 4u; byte++) {
        let amount = (value >> (8u * byte)) & 255u;
        sum = add_lanes(sum, amount << (8u * (byte % distance)));
    }
    return sum;
}

@compute @workgroup_size(64)
fn delta_local(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let tile = group.y * parameters.tile_groups_x + group.x;
    if tile >= tile_count() { return; }
    let distance = 1u << (parameters.kind - 2u);
    let start = tile * 1024u + lane * 16u;
    let end = min(start + 16u, words());
    var sum = 0u;
    for (var index = start; index < end; index++) {
        sum = add_lanes(sum, residue_sum(source[index], distance));
    }
    totals[lane] = sum;
    workgroupBarrier();
    for (var stride = 1u; stride < 64u; stride *= 2u) {
        var previous = 0u;
        if lane >= stride { previous = totals[lane - stride]; }
        workgroupBarrier();
        totals[lane] = add_lanes(totals[lane], previous);
        workgroupBarrier();
    }
    if lane == 63u { tiles[tile] = totals[lane]; }
    var carry = 0u;
    if lane > 0u { carry = totals[lane - 1u]; }
    for (var index = start; index < end; index++) {
        let value = source[index];
        var packed = 0u;
        for (var byte = 0u; byte < 4u; byte++) {
            let shift = 8u * (byte % distance);
            carry = add_lanes(carry, ((value >> (8u * byte)) & 255u) << shift);
            packed |= ((carry >> shift) & 255u) << (8u * byte);
        }
        result[index] = packed;
    }
}

@compute @workgroup_size(1)
fn delta_carries() {
    var carry = 0u;
    for (var tile = 0u; tile < tile_count(); tile++) {
        let next = add_lanes(carry, tiles[tile]);
        tiles[tile] = carry;
        carry = next;
    }
}

@compute @workgroup_size(64)
fn delta_finish(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let index = (group.y * parameters.word_groups_x + group.x) * 64u + lane;
    if index >= words() { return; }
    var carry = tiles[index / 1024u];
    if parameters.kind == 2u { carry = (carry & 255u) * 0x01010101u; }
    if parameters.kind == 3u { carry = (carry & 65535u) * 0x00010001u; }
    result[index] = add_lanes(result[index], carry);
}
