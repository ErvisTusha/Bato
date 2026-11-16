"""Measure Bato file-piece factoring as a research wrapper."""
import argparse
import csv
import hashlib
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bato_tools.resources import available_cpu_count, memory_state, resolve_memory_budget

parser = argparse.ArgumentParser(description="Benchmark standalone Bato files with shared fragments.")
parser.add_argument("--branch", type=Path, required=True)
parser.add_argument("--helper", type=Path, required=True)
parser.add_argument("--input", type=Path, action="append", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--variants", nargs="+", default=["stock", "adaptive"],
                    metavar="STOCK_ADAPTIVE_SWEEP_OR_BYTES")
parser.add_argument("--repetitions", type=int, default=2)
parser.add_argument("--memory-budget", type=int)
parser.add_argument("--timeout", type=int, default=3600, help="Per-command timeout in seconds")
parser.add_argument("--fragment-memory-budget", type=int,
                    help="Limit the fragment helper's memory allowance independently")
args = parser.parse_args()
budgets = [args.memory_budget, args.fragment_memory_budget]
if args.repetitions < 1 or args.timeout < 1 or any(value is not None and value < 1 for value in budgets):
    parser.error("repetitions, timeout and memory budgets must be positive")
binary, helper = (path.resolve(strict=True) for path in (args.branch, args.helper))
inputs = [path.resolve(strict=True) for path in args.input]
output_path = args.output.resolve()
if output_path in inputs or any(not path.is_file() for path in inputs):
    parser.error("each input must be a file distinct from the output")
piece_sizes = [str(1 << exponent) for exponent in range(11, 24)]
variants = []
for variant in args.variants:
    if variant == "sweep":
        variants.extend(piece_sizes)
    elif variant in ("stock", "adaptive") or variant in piece_sizes:
        variants.append(variant)
    else:
        parser.error(f"invalid variant {variant!r}")
if len(variants) != len(set(variants)):
    parser.error("variants must be unique")
available = memory_state()["available_ram_bytes"]
budget = min(available, resolve_memory_budget(args.memory_budget))
if budget < 1:
    parser.error("no available host memory")
workers = min(available_cpu_count(), 256)
fragment_budget = min(budget, args.fragment_memory_budget or budget)
# The window holds original bytes, literals, and a capped fingerprint index.
window_capacity = min(1 << 30, fragment_budget // 16)
options = ["--threads", str(workers), "--memory-budget", str(budget), "--compute", "cpu"]
tag = b"BTF2\x00"
legacy_tag = b"BTF1\x00"

def power_of_two_at_most(value):
    return 1 << (value.bit_length() - 1) if value > 0 else 0

def plan(original, variant):
    window = min(power_of_two_at_most(window_capacity), 1 << (max(1, original) - 1).bit_length())
    if variant == "adaptive":
        pieces = max(4, min(16, workers // 2))
        size = power_of_two_at_most(min(original, window) // pieces)
    else:
        size = int(variant)
    return size, min(window, size * 4096)


def cpu():
    runs = [resource.getrusage(k) for k in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN)]
    return sum(x.ru_utime + x.ru_stime for x in runs)

def run(cmd, *arguments):
    subprocess.run([cmd, *arguments, *(options if cmd == binary else [])], check=True, timeout=args.timeout,
                   stdout=subprocess.DEVNULL)

def pack(source, archive, scratch, variant):
    original = source.stat().st_size
    if variant == "stock" or (variant == "adaptive" and original < 128 * 1024):
        run(binary, "-c", source, "-o", archive)
        return 0, 0, "stock"
    size, window = plan(original, variant)
    if size >= original:
        return size, window, "single"
    if size < 2048 or window <= size:
        if variant == "adaptive":
            run(binary, "-c", source, "-o", archive)
            return size, window, "stock"
        return size, window, "outside working window"
    reduced = scratch / "compacted"
    prepared = subprocess.check_output(
        [helper, "prepare-file", source, reduced, str(size), str(window)], timeout=args.timeout, text=True,
    ).strip()
    if prepared not in ("0", "1"):
        raise RuntimeError("unexpected fragment helper response")
    if prepared == "0" or (variant == "adaptive"
                            and sum(p.stat().st_size for p in reduced.rglob("*") if p.is_file()) > original // 2):
        if variant == "adaptive":
            run(binary, "-c", source, "-o", archive)
            return size, window, "stock"
        return size, window, "no references"
    native = scratch / "native.bato"
    segment = min(256 * 1024, max(64 * 1024, size // 2)) if variant == "adaptive" else 256 * 1024
    run(binary, "archive", reduced, "-o", native,
        "--segment-size", str(segment), "--chunk-size", str(segment))
    with archive.open("wb") as output, native.open("rb") as disk:
        output.write(tag)
        output.write(window.to_bytes(8, "little"))
        shutil.copyfileobj(disk, output)
    return size, window, "wrapped"

def restore(archive, output, scratch):
    with archive.open("rb") as data:
        marker = data.read(len(tag))
        if marker in (tag, legacy_tag):
            if marker == tag:
                field = data.read(8)
                window = int.from_bytes(field, "little")
                if len(field) != 8 or not 0 < window <= window_capacity:
                    raise ValueError("fragment window exceeds the available memory allowance")
            native = scratch / "restore-native.bato"
            with native.open("wb") as disk:
                shutil.copyfileobj(data, disk)
            decoded = scratch / "decoded"
            run(binary, "extract", native, "-o", decoded)
            if marker == tag:
                run(helper, "restore-file", decoded, output, str(window))
            else:
                pieces = scratch / "restore-pieces"
                run(helper, "restore", decoded, pieces)
                with output.open("wb") as raw:
                    for index, piece in enumerate(sorted(pieces.iterdir())):
                        if not piece.is_file() or piece.name != f"{index:08d}":
                            raise ValueError("invalid restored piece sequence")
                        with piece.open("rb") as disk:
                            shutil.copyfileobj(disk, raw)
            return
    run(binary, "-d", archive, "-o", output)

def fingerprint(path):
    with path.open("rb") as data:
        return hashlib.file_digest(data, "sha256").hexdigest()

output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open("w") as output:
    rows = csv.writer(output, delimiter="\t")
    rows.writerow(["input","rep","variant","piece","window","wrapped","bytes","pack_wall","pack_cpu","restore_wall","restore_cpu","digest"])
    for i, source in enumerate(inputs):
        name = source.name
        for rep in range(args.repetitions):
            shift = (i + rep) % len(variants)
            for variant in variants[shift:] + variants[:shift]:
                with tempfile.TemporaryDirectory(dir=output_path.parent) as tmp:
                    scratch = Path(tmp)
                    archive, restored = (scratch / x for x in ("out.bato", "restored"))
                    start, used = time.monotonic(), cpu()
                    size, window, outcome = pack(source, archive, scratch, variant)
                    pw, pc = time.monotonic() - start, cpu() - used
                    if outcome in ("single", "no references", "outside working window"):
                        rows.writerow([name,rep,variant,size,window,outcome,"",pw,pc,"","",""])
                        output.flush()
                        continue
                    start, used = time.monotonic(), cpu()
                    restore(archive, restored, scratch)
                    dw, dc = time.monotonic() - start, cpu() - used
                    subprocess.run(["cmp","-s",source,restored],check=True)
                    rows.writerow([name,rep,variant,size,window,outcome,archive.stat().st_size,pw,pc,dw,dc,fingerprint(archive)])
                    output.flush()
        print(f"{name} verified", flush=True)
