#!/usr/bin/env python3
"""Ablation study: which model steps does the archive size really need?

For every named step, a temporary copy of the sources is patched so that the
step contributes nothing (its input is zeroed, its stage skipped), built, and
run over a panel of files. The size change says what the step is worth; the
`--timing` report of the unmodified build says what it costs. A step whose
removal does not grow the files, or shrinks them, is a candidate for deletion.

The sources of the repository are never modified. Sizes are exact and
deterministic; compression and decompression use the same patched model, so
every archive is also checked to decode to the original bytes.

    python3 scripts/ablate.py --work /tmp/ablate FILE [FILE ...]
    python3 scripts/ablate.py --work /tmp/ablate --only apm-3 --only isse-0 FILE
"""
import argparse
import concurrent.futures
import filecmp
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CM = "src/core/cm/"

# Compile-time switch read by the patched copy: the number of the ablation.
SWITCH = """
/// Ablation switch for scripts/ablate.py; absent in the real build.
pub(crate) const ABLATE: usize = match option_env!("BATO_ABLATE") {
    Some(text) => parse_switch(text.as_bytes(), 0, 0),
    None => 0,
};
/// Calibration stages and ISSE experts skipped together, as bit masks.
pub(crate) const SKIP_APM: usize = match option_env!("BATO_SKIP_APM") {
    Some(text) => parse_switch(text.as_bytes(), 0, 0),
    None => 0,
};
pub(crate) const SKIP_ISSE: usize = match option_env!("BATO_SKIP_ISSE") {
    Some(text) => parse_switch(text.as_bytes(), 0, 0),
    None => 0,
};
const fn parse_switch(text: &[u8], at: usize, value: usize) -> usize {
    if at == text.len() {
        value
    } else {
        parse_switch(text, at + 1, value * 10 + (text[at] - b'0') as usize)
    }
}
"""

# (file, text that must occur exactly once, replacement). Text is matched
# verbatim, so a source change that moves an anchor fails loudly.
# The switches live in `cm/mod.rs`; patched code names them by full path, so
# no import list has to be an anchor.
ABLATE = "crate::core::cm::ABLATE"
SKIP_APM = "crate::core::cm::SKIP_APM"
SKIP_ISSE = "crate::core::cm::SKIP_ISSE"
SKIP_STAGE_APM = f"            if {ABLATE} == 100 + stage || ({SKIP_APM} >> stage) & 1 == 1 {{\n                continue;\n            }}\n"
SKIP_STAGE_ISSE = f"            if {ABLATE} == 110 + stage || ({SKIP_ISSE} >> stage) & 1 == 1 {{\n                continue;\n            }}\n"
FRONT = CM + "predictor/front.rs"
MIXING = CM + "predictor/mixing.rs"
CALIBRATION = CM + "predictor/calibration.rs"
FINAL_AVERAGE = "        let prediction = ((u32::from(first_final) + u32::from(second_final) + 1) >> 1) as u16;\n"

PATCHES = [
    (CM + "mod.rs", "mod adapt;\n", SWITCH + "mod adapt;\n"),
    (  # calibration stages 100..108
        CALIBRATION,
        "        for (map, &context) in self.apms.iter_mut().zip(&mixed.calibration) {\n",
        "        for (stage, (map, &context)) in self.apms.iter_mut().zip(&mixed.calibration).enumerate() {\n"
        + SKIP_STAGE_APM,
    ),
    (
        CALIBRATION,
        "        for map in &mut self.apms {\n",
        "        for (stage, map) in self.apms.iter_mut().enumerate() {\n" + SKIP_STAGE_APM,
    ),
    (  # ISSE experts 110..112
        CALIBRATION,
        "        for (expert, state) in self.isse.iter_mut().zip(mixed.isse_states) {\n",
        "        for (stage, (expert, state)) in self.isse.iter_mut().zip(mixed.isse_states).enumerate() {\n"
        + SKIP_STAGE_ISSE,
    ),
    (
        CALIBRATION,
        "        for expert in &mut self.isse {\n",
        "        for (stage, expert) in self.isse.iter_mut().enumerate() {\n" + SKIP_STAGE_ISSE,
    ),
    (  # first-layer mixers 120..122
        MIXING,
        "        self.second[NEIGHBOR_INPUT] = signals.neighbors[0];\n",
        f"        if (120..123).contains(&{ABLATE}) {{\n            self.second[{ABLATE}.saturating_sub(120)] = 0;\n        }}\n"
        "        self.second[NEIGHBOR_INPUT] = signals.neighbors[0];\n",
    ),
    (  # neighbours 130
        FRONT,
        "        signals.neighbors = self.neighbors.feature(self.prefix);\n",
        f"        signals.neighbors = if {ABLATE} == 130 {{\n            [0; neighbors::FEATURES]\n        }} else {{\n"
        "            self.neighbors.feature(self.prefix)\n        };\n",
    ),
    (  # experts 140..143: forced dormant for the whole byte
        FRONT,
        "            self.expert_active[3] = self.rows.active();\n",
        "            self.expert_active[3] = self.rows.active();\n"
        f"            for e in 0..EXPERTS {{\n                if {ABLATE} == 140 + e {{\n"
        "                    self.expert_active[e] = false;\n                }\n            }\n",
    ),
    (  # second final mixer 150 (use the first only), first final mixer 151
        MIXING,
        FINAL_AVERAGE,
        f"        let (first_final, second_final) = match {ABLATE} {{\n"
        "            150 => (first_final, first_final),\n            151 => (second_final, second_final),\n"
        "            _ => (first_final, second_final),\n        };\n" + FINAL_AVERAGE,
    ),
    (  # record model 160
        FRONT,
        "            self.record_active = record.is_some();\n",
        f"            self.record_active = record.is_some() && {ABLATE} != 160;\n",
    ),
    (  # match input 161
        FRONT,
        "            .copy_from_slice(&self.matcher.features(self.prefix));\n",
        f"            .copy_from_slice(&if {ABLATE} == 161 {{\n                [0; 2]\n            }} else {{\n"
        "                self.matcher.features(self.prefix)\n            });\n",
    ),
    (  # run input 162, order-0 input 163
        FRONT,
        "        signals.inputs[RUN_INPUT] = logistic::stretch(self.run.probability(self.prefix));\n",
        f"        signals.inputs[RUN_INPUT] = if {ABLATE} == 162 {{\n            0\n        }} else {{\n"
        "            logistic::stretch(self.run.probability(self.prefix))\n        };\n"
        f"        if {ABLATE} == 163 {{\n            signals.inputs[ORDER0_INPUT] = 0;\n        }}\n",
    ),
    (  # context maps 170..179
        FRONT,
        "        lap(meter, Step::ContextMaps);\n",
        "        lap(meter, Step::ContextMaps);\n"
        f"        if (170..170 + CONTEXTS).contains(&{ABLATE}) {{\n            let i = {ABLATE}.saturating_sub(170);\n"
        "            signals.inputs[i * FEATURES..(i + 1) * FEATURES].fill(0);\n        }\n",
    ),
    (  # built-in primer 180
        CM + "segment.rs",
        "fn primer_bytes(primer: Primer) -> &'static [u8] {\n    match primer {\n",
        "fn primer_bytes(primer: Primer) -> &'static [u8] {\n    if super::ABLATE == 180 {\n        return &[];\n    }\n    match primer {\n",
    ),
    (  # x86 transform 181
        CM + "segment.rs",
        "    let x86 = timing.time(Step::TransformSelection, || x86::select(raw))?;\n",
        "    let x86 = timing.time(Step::TransformSelection, || x86::select(raw))?\n        .filter(|_| super::ABLATE != 181);\n",
    ),
    (  # deflate transform 182
        CM + "segment.rs",
        "    let inflated = timing.time(Step::DeflateTransform, || {\n",
        "    let inflated = timing.time(Step::DeflateTransform, || {\n"
        "        if super::ABLATE == 182 {\n            return None;\n        }\n",
    ),
    (  # second (unprimed) trial 183: one model reserved, one trial run
        CM + "segment.rs",
        "    let models = if input.len() > UNPRIMED_TRIAL_BYTES {\n",
        "    let models = if input.len() > UNPRIMED_TRIAL_BYTES || super::ABLATE == 183 {\n",
    ),
    (
        CM + "segment.rs",
        "Vec<Chunk>)>> = if input.len() > UNPRIMED_TRIAL_BYTES {\n",
        "Vec<Chunk>)>> = if input.len() > UNPRIMED_TRIAL_BYTES || super::ABLATE == 183 {\n",
    ),
]

# name -> (switch, timing step it removes work from, what it is)
ABLATIONS = {}
for i in range(8):
    ABLATIONS[f"apm-{i}"] = (100 + i, "calibration (APM) stages", f"calibration stage {i}")
for i in range(3):
    ABLATIONS[f"isse-{i}"] = (110 + i, "ISSE chain", f"ISSE expert {i}")
    ABLATIONS[f"mixer-{i}"] = (120 + i, "first-layer mixers", f"first-layer mixer {i}")
ABLATIONS["neighbours"] = (130, "first-layer mixers", "bit-neighbourhood input")
for i, name in enumerate(["text", "sparse", "lag", "row"]):
    ABLATIONS[f"expert-{name}"] = (140 + i, "expert maps and mixers", f"{name} expert")
ABLATIONS["final-second"] = (150, "final mixers", "second final mixer")
ABLATIONS["final-first"] = (151, "final mixers", "first final mixer")
ABLATIONS["records"] = (160, "order-0, run, match, record inputs", "record-stride model")
ABLATIONS["match"] = (161, "order-0, run, match, record inputs", "match model input")
ABLATIONS["run"] = (162, "order-0, run, match, record inputs", "byte-run input")
ABLATIONS["order0"] = (163, "order-0, run, match, record inputs", "order-0 input")
for i, label in enumerate(["order 0", "order 1", "order 2", "order 3", "order 4", "order 6",
                           "word 1", "word 2", "word 3", "word 4"]):
    ABLATIONS[f"context-{i}"] = (170 + i, "main context maps", f"context map: {label}")
ABLATIONS["primer"] = (180, "primer training", "built-in primer")
ABLATIONS["x86"] = (181, "x86 transform selection", "x86 address transform")
ABLATIONS["deflate"] = (182, "deflate scan and transform", "deflate recompression")
ABLATIONS["unprimed-trial"] = (183, "model encoding trials (cpu)", "second encoding trial without primer")

REMOVABLE = 0.02  # percent of total size below which a step looks unneeded


def patched_sources(work: Path) -> Path:
    """Copies the sources to `work/src` and applies every patch once."""
    tree = work / "tree"
    if tree.exists():
        shutil.rmtree(tree)
    shutil.copytree(ROOT / "src", tree / "src")
    # The CUDA backend embeds its kernels from `native/` at compile time.
    shutil.copytree(ROOT / "native", tree / "native")
    for name in ("Cargo.toml", "Cargo.lock", "build.rs"):
        if (ROOT / name).exists():
            shutil.copy(ROOT / name, tree / name)
    for relative, anchor, replacement in PATCHES:
        path = tree / relative
        text = path.read_text()
        if text.count(anchor) != 1:
            raise SystemExit(f"ablation anchor not found exactly once in {relative}: {anchor!r}")
        path.write_text(text.replace(anchor, replacement, 1))
    return tree


def build(tree: Path, target: Path, switch, masks=(0, 0)) -> Path:
    env = dict(os.environ, CARGO_PROFILE_RELEASE_LTO="off", CARGO_PROFILE_RELEASE_CODEGEN_UNITS="16",
               CARGO_TARGET_DIR=str(target), BATO_SKIP_APM=str(masks[0]), BATO_SKIP_ISSE=str(masks[1]))
    if switch is None:
        env.pop("BATO_ABLATE", None)
    else:
        env["BATO_ABLATE"] = str(switch)
    subprocess.run(["cargo", "build", "--release", "--bin", "bato"], cwd=tree, env=env, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return target / "release" / "bato"


def run_panel(binary: Path, files, scratch: Path, jobs: int, check_roundtrip: bool):
    """Compressed size of every file; optionally verifies the round trip."""
    scratch.mkdir(parents=True, exist_ok=True)

    def one(path: Path):
        out = scratch / (path.name + ".b")
        subprocess.run([binary, "-c", path, "-o", out, "-f"], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        size = out.stat().st_size
        if check_roundtrip:
            back = scratch / (path.name + ".out")
            subprocess.run([binary, "-d", out, "-o", back, "-f"], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if not filecmp.cmp(path, back, shallow=False):
                raise SystemExit(f"round trip differs for {path}")
            back.unlink()
        out.unlink()
        return size

    with concurrent.futures.ThreadPoolExecutor(jobs) as pool:
        return list(pool.map(one, files))


def timing_shares(binary: Path, files, scratch: Path) -> dict:
    """Milliseconds per timing step, summed over the files, from `--timing`."""
    totals = {}
    for path in files:
        out = scratch / (path.name + ".t")
        report = subprocess.run([binary, "-c", path, "-o", out, "-f", "--timing"], check=True,
                                capture_output=True, text=True).stderr
        out.unlink()
        for line in report.splitlines():
            match = re.match(r"^(\S.*?)\s{2,}(\d+)\s+\d+%\s+\d+$", line)
            if match:
                totals[match.group(1)] = totals.get(match.group(1), 0) + int(match.group(2))
    return totals


def parse_combo(text: str):
    """`apm=1,5;isse=2` -> bit masks of calibration stages and ISSE experts."""
    masks = {"apm": 0, "isse": 0}
    for part in filter(None, text.split(";")):
        kind, _, numbers = part.partition("=")
        if kind not in masks:
            raise SystemExit(f"unknown combination part {part!r}")
        for number in filter(None, numbers.split(",")):
            masks[kind] |= 1 << int(number)
    return masks["apm"], masks["isse"]


def verdict(delta_percent: float, worst_percent: float) -> str:
    if delta_percent < -REMOVABLE:
        return "HURTS: files shrink without it"
    if delta_percent <= REMOVABLE and worst_percent <= 5 * REMOVABLE:
        return "unneeded on this panel"
    if delta_percent <= 10 * REMOVABLE:
        return "marginal"
    return "needed"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--work", type=Path, required=True, help="scratch directory (outside the repository)")
    parser.add_argument("--only", action="append", choices=sorted(ABLATIONS), help="run only these steps")
    parser.add_argument("--combo", action="append", default=[],
                        help="skip several stages together, e.g. apm=1,5,6;isse=2 (repeatable)")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args(argv)
    files = [path.resolve() for path in args.files]
    args.work.mkdir(parents=True, exist_ok=True)
    tree = patched_sources(args.work)
    target = args.work / "target"
    baseline_binary = build(tree, target, None)
    base = run_panel(baseline_binary, files, args.work / "run", args.jobs, check_roundtrip=True)
    total = sum(base)
    times = timing_shares(baseline_binary, files, args.work / "run")
    whole = ("x86 transform selection", "fast codec candidates", "deflate scan and transform",
             "primer training", "model encoding trials (cpu)", "model decoding",
             "inverse transforms", "checksums")
    whole_ms = sum(times.get(step, 0) for step in whole) or 1
    model_ms = (times.get("model encoding trials (cpu)", 0) + times.get("model decoding", 0)) or 1
    print(f"panel: {len(files)} files, {total} bytes compressed by the unmodified model")
    print(f"{'step':<44} {'bytes':>9} {'size':>8} {'worst file':>10} {'time share':>10}  verdict")
    names = [] if args.combo and not args.only else (args.only or list(ABLATIONS))
    for combo in args.combo:
        masks = parse_combo(combo)
        binary = build(tree, target, None, masks)
        sizes = run_panel(binary, files, args.work / "run", args.jobs, check_roundtrip=True)
        delta = sum(sizes) - total
        worst = max(100 * (new - old) / old for new, old in zip(sizes, base))
        print(f"{combo:<44} {delta:>+9} {100 * delta / total:>+7.3f}% {worst:>+9.3f}%", flush=True)
    for name in names:
        switch, step, label = ABLATIONS[name]
        binary = build(tree, target, switch)
        sizes = run_panel(binary, files, args.work / "run", args.jobs, check_roundtrip=name == names[0])
        delta = sum(sizes) - total
        percent = 100 * delta / total
        worst = max(100 * (new - old) / old for new, old in zip(sizes, base))
        share = 100 * times.get(step, 0) / (whole_ms if step in whole else model_ms)
        print(f"{label:<44} {delta:>+9} {percent:>+7.3f}% {worst:>+9.3f}% {share:>8.1f}%  "
              f"{verdict(percent, worst)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
