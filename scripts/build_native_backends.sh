#!/usr/bin/env bash
# Build optional native vendor accelerators. The normal Rust build requires no
# ROCm, oneAPI or Apple SDK. Each backend is an independent shared library.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${BATO_NATIVE_OUT:-$ROOT/target/native}"
mkdir -p "$OUT"

usage() {
  echo "Usage: $0 hip|level-zero|metal|all" >&2
  exit 2
}

build_hip() {
  command -v hipcc >/dev/null || { echo "hipcc (ROCm/HIP SDK) is required" >&2; exit 1; }
  echo "Building AMD HIP plugin"
  hipcc -O3 -std=c++17 -fPIC -shared "$ROOT/native/hip/filters.hip" \
    -o "$OUT/libbato_hip.so"
}

build_level_zero() {
  command -v icpx >/dev/null || { echo "icpx (Intel oneAPI DPC++ SDK) is required" >&2; exit 1; }
  echo "Building Intel oneAPI Level Zero plugin"
  icpx -fsycl -fsycl-targets=spir64 -O3 -std=c++17 -fPIC -shared \
    "$ROOT/native/level_zero/filters.cpp" -o "$OUT/libbato_level_zero.so"
}

build_metal() {
  [[ "$(uname -s)" == "Darwin" ]] || { echo "Metal plugin requires macOS" >&2; exit 1; }
  command -v xcrun >/dev/null || { echo "Xcode command line tools are required" >&2; exit 1; }
  echo "Building Apple Silicon native Metal plugin"
  xcrun clang++ -O3 -std=c++17 -fobjc-arc -dynamiclib \
    -framework Foundation -framework Metal \
    "$ROOT/native/metal/filters.mm" -o "$OUT/libbato_metal.dylib"
}

case "${1:-}" in
  hip) build_hip ;;
  level-zero) build_level_zero ;;
  metal) build_metal ;;
  all)
    case "$(uname -s)" in
      Darwin) build_metal ;;
      *) build_hip; build_level_zero ;;
    esac
    ;;
  *) usage ;;
esac

echo "Native libraries in $OUT"
