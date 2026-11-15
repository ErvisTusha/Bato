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
  # Multiple GPU target code objects can live in the same shared library;
  # specify BATO_HIP_OFFLOAD_ARCHS=gfx1100,gfx1101,... on SDK build hosts.
  # Unsupported architectures must fail the build rather than being silently
  # omitted from a release advertised as all-native.
  local -a flags=(-O3 -std=c++17 -fPIC -shared -x hip)
  if [[ -n "${BATO_HIP_OFFLOAD_ARCHS:-}" ]]; then
    local -a targets
    IFS=',; ' read -ra targets <<< "$BATO_HIP_OFFLOAD_ARCHS"
    local arch
    for arch in "${targets[@]}"; do
      [[ -n "$arch" ]] || continue
      [[ "$arch" =~ ^gfx[0-9a-f]+$ ]] || {
        echo "Invalid HIP offload target: $arch" >&2
        exit 2
      }
      flags+=("--offload-arch=$arch")
    done
  fi
  hipcc "${flags[@]}" "$ROOT/native/hip/filters.hip" \
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
