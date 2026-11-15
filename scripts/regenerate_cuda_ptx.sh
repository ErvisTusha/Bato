#!/usr/bin/env bash
# Regenerate the committed PTX from the unmodified CUDA kernel source.
# This is an explicit maintainer command; normal Cargo builds load the
# pre-generated PTX and do not need NVIDIA's CUDA Toolkit.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIR="$ROOT/native/cuda"
command -v clang >/dev/null || { echo "clang with NVPTX target is required" >&2; exit 1; }

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

(
    cd "$DIR"
    clang -x cuda --cuda-device-only -nocudainc -nocudalib \
        --cuda-gpu-arch=sm_75 -O3 -S -o "$tmp/body.ptx" kernels.cu
)
if command -v sha256sum >/dev/null; then
    sha="$(sha256sum "$DIR/kernels.cu" | cut -d' ' -f1)"
else
    sha="$(shasum -a 256 "$DIR/kernels.cu" | cut -d' ' -f1)"
fi

{
    printf '// source sha256: %s\n' "$sha"
    cat "$tmp/body.ptx"
} > "$tmp/kernels.ptx"

cp "$tmp/kernels.ptx" "$DIR/kernels.ptx"
echo "Regenerated $DIR/kernels.ptx from kernels.cu ($sha)"
echo "Run cargo test --features cuda ptx_is_generated_from_the_current_source"
