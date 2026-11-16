import argparse
import csv
import hashlib
from pathlib import Path
import resource
import shutil
import subprocess
import tempfile
import time

parser = argparse.ArgumentParser(description="Compare native Bato with the cross-file fragment prototype.")
for name in ("main", "branch", "helper", "output"):
    parser.add_argument(f"--{name}", type=Path, required=True)
parser.add_argument("--input", type=Path, action="append", required=True)
parser.add_argument("--variants", nargs="+", choices=["main", "branch", "solid", "wide", "narrow", "512k", "1m"],
                    default=["main", "branch", "solid"])
parser.add_argument("--repetitions", type=int, default=3)
parser.add_argument("--memory-budget", type=int, default=7730941132)
args = parser.parse_args()
if args.repetitions < 1 or args.memory_budget < 1 or len(set(args.variants)) != len(args.variants):
    parser.error("repetitions and memory budget must be positive, with unique variants")
main, branch, helper = (binary.resolve(strict=True) for binary in (args.main, args.branch, args.helper))
output = args.output.resolve()
if any(output.is_relative_to(path.resolve()) for path in args.input):
    parser.error("output must be outside the input trees")
options = ["--threads", "0", "--memory-budget", str(args.memory_budget)]
geometry = {"solid": 262144, "wide": 131072, "narrow": 65536, "512k": 524288, "1m": 1048576}
tag = b"BTD1\x00"

def inventory(tree):
    entries = {}
    for p in sorted(tree.rglob("*")):
        name = p.relative_to(tree)
        entries[name] = None if p.is_dir() else hashlib.sha256(p.read_bytes()).digest()
    return entries

def cpu():
    stat = [resource.getrusage(scope) for scope in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN)]
    return sum(s.ru_utime + s.ru_stime for s in stat)

def call(binary, *arguments):
    extra = [] if binary == helper else options + ([] if binary == main else ["--compute", "cpu"])
    subprocess.run([binary, *arguments, *extra], stdout=subprocess.DEVNULL, check=True, timeout=240)

def wrapped(source, target):
    with source.open("rb") as archive, target.open("wb") as output:
        output.write(tag)
        shutil.copyfileobj(archive, output)

def pack(variant, source, tmp):
    binary = main if variant == "main" else branch
    out = tmp / "archive.bato"
    saved = 0
    started = time.monotonic()
    previous = cpu()
    if variant in geometry:
        staged = tmp / "compact"
        call(helper, "prepare", source, staged)
        if (staged / "recipes").exists():
            native = tmp / "compact.bato"
            size = str(geometry[variant])
            call(binary, "archive", staged, "-o", native, "--segment-size", size, "--chunk-size", str(min(int(size), 262144)))
            wrapped(native, out)
            saved = 1
        else:
            call(binary, "archive", source, "-o", out)
    else:
        call(binary, "archive", source, "-o", out)
    return out, saved, time.monotonic() - started, cpu() - previous

def restore(variant, archive, tmp):
    binary = main if variant == "main" else branch
    out = tmp / "restored"
    started = time.monotonic()
    previous = cpu()
    if variant in geometry:
        with archive.open("rb") as source:
            if source.read(len(tag)) == tag:
                native = tmp / "wrapped.bato"
                with native.open("wb") as output:
                    shutil.copyfileobj(source, output)
                decoded = tmp / "decoded"
                call(binary, "extract", native, "-o", decoded)
                call(helper, "restore", decoded, out)
            else:
                call(binary, "extract", archive, "-o", out)
    else:
        call(binary, "extract", archive, "-o", out)
    return out, time.monotonic() - started, cpu() - previous

output.parent.mkdir(parents=True, exist_ok=True)
with output.open("w") as log:
    rows = csv.writer(log, delimiter="\t")
    rows.writerow(["input", "rep", "variant", "bytes", "pack_wall", "pack_cpu", "restore_wall",
                   "restore_cpu", "wrapped", "digest"])
    for i, source in enumerate(path.resolve(strict=True) for path in args.input):
        name = source.name
        expected = inventory(source)
        for rep in range(args.repetitions):
            order = args.variants
            shift = (i + rep) % len(order)
            for variant in order[shift:] + order[:shift]:
                with tempfile.TemporaryDirectory(dir=output.parent) as folder:
                    tmp = Path(folder)
                    archive, active, pack_wall, pack_cpu = pack(variant, source, tmp)
                    result, restore_wall, restore_cpu = restore(variant, archive, tmp)
                    if inventory(result) != expected:
                        raise RuntimeError(f"{name} failed restoration verification with {variant}")
                    for name_in_tree, digest in expected.items():
                        if digest is not None:
                            subprocess.run(["cmp", "-s", source / name_in_tree, result / name_in_tree], check=True)
                    rows.writerow([name, rep, variant, archive.stat().st_size, pack_wall, pack_cpu, restore_wall,
                                   restore_cpu, active, hashlib.sha256(archive.read_bytes()).hexdigest()])
                    log.flush()
        print(f"{name} verified", flush=True)
