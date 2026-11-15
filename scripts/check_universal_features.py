#!/usr/bin/env python3
"""Static checks for Bato's one-executable-per-OS universal GPU defaults.

The SDK-built HIP / Level Zero libraries are embedded only when supplied at
build time. The Rust-native CUDA, wgpu and CPU fallback paths are always
included by a default build, with all optional vendor plugin loaders compiled.
"""
from pathlib import Path
import re
import sys
import tomllib

ROOT = Path(__file__).resolve().parent.parent
manifest = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))
features = manifest["features"]
expected = {"cuda", "hip", "level-zero", "metal", "wgpu"}
actual = set(features.get("default", []))
if actual != expected:
    raise SystemExit(f"Bato universal default GPU features mismatch: {actual} != {expected}")

for vendor in expected:
    if vendor not in features:
        raise SystemExit(f"Missing GPU Cargo feature: {vendor}")

compute = (ROOT / "src/core/types.rs").read_text(encoding="utf-8")
if not re.search(r"#\[default\]\s*Auto\s*,", compute):
    raise SystemExit("Default compute mode must be Auto, not CPU or one GPU vendor")

source = (ROOT / "src/main.rs").read_text(encoding="utf-8")
for name in ("auto", "cpu", "cuda", "hip", "level-zero", "metal", "wgpu"):
    if f'Some("{name}") => Compute::' not in source:
        raise SystemExit(f"Missing CLI GPU compute option: {name}")

must_exist = (
    "src/core/gpu/backend/cuda/mod.rs",
    "src/core/gpu/backend/wgpu/mod.rs",
    "src/core/gpu/backend/native.rs",
    "src/core/gpu/scheduler.rs",
    "src/core/gpu/operations/filters.rs",
    "native/cuda/kernels.ptx",
    "native/hip/filters.hip",
    "native/level_zero/filters.cpp",
    "native/metal/filters.mm",
    "build.rs",
)
for item in must_exist:
    if not (ROOT / item).is_file():
        raise SystemExit(f"Missing universal backend component: {item}")

print("Universal GPU features: CUDA, HIP, Level Zero, Metal, wgpu; CPU fallback and Auto default: OK")
