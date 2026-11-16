"""Measure Bato file-piece factoring as a research wrapper."""
import argparse
import csv
import hashlib
from pathlib import Path
import resource
import shutil
import subprocess
import tempfile
import time

parser = argparse.ArgumentParser(description="Benchmark standalone Bato files with shared fragments.")
parser.add_argument("--branch", type=Path, required=True)
parser.add_argument("--helper", type=Path, required=True)
parser.add_argument("--input", type=Path, action="append", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--variants", nargs="+", default=["stock", "adaptive"],
                    metavar="STOCK_ADAPTIVE_SWEEP_OR_BYTES")
parser.add_argument("--repetitions", type=int, default=2)
parser.add_argument("--memory-budget", type=int, default=7730941132)
args = parser.parse_args()
if args.repetitions < 1 or args.memory_budget < 1:
    parser.error("repetitions and memory budget must be positive")
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
options = ["--threads", "0", "--memory-budget", str(args.memory_budget), "--compute", "cpu"]
tag = b"BTF1\x00"

def cpu():
    runs = [resource.getrusage(k) for k in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN)]
    return sum(x.ru_utime + x.ru_stime for x in runs)

def run(cmd, *args):
    subprocess.run([cmd, *args, *(options if cmd == binary else [])], check=True, timeout=300,
                   stdout=subprocess.DEVNULL)

def pack(source, archive, scratch, variant):
    stored = source.stat().st_size
    adaptive = variant == "adaptive" and 128 * 1024 <= stored <= 8 * 1024 * 1024
    if variant == "stock" or (variant == "adaptive" and not adaptive):
        run(binary, "-c", source, "-o", archive)
        return 0, "stock"
    if variant != "stock":
        size = 1 << ((stored // 4).bit_length() - 1) if adaptive else int(variant)
        if stored > 8 * 1024 * 1024:
            return size, "outside prototype limit"
        if size >= stored:
            return size, "single"
        pieces = scratch / "pieces"
        pieces.mkdir()
        with source.open("rb") as raw:
            counter = 0
            while chunk := raw.read(size):
                (pieces / f"{counter:08}").write_bytes(chunk)
                counter += 1
        reduced = scratch / "compacted"
        run(helper, "prepare", pieces, reduced)
        if (reduced / "recipes").exists():
            native = scratch / "native.bato"
            segment = min(256 * 1024, max(64 * 1024, size // 2)) if adaptive else 256 * 1024
            if adaptive and sum(f.stat().st_size for f in reduced.rglob("*") if f.is_file()) > stored // 2:
                run(binary, "-c", source, "-o", archive)
                return size, "stock"
            run(binary, "archive", reduced, "-o", native,
                "--segment-size", str(segment), "--chunk-size", str(segment))
            with archive.open("wb") as output, native.open("rb") as disk:
                output.write(tag)
                shutil.copyfileobj(disk, output)
            return size, "wrapped"
        if adaptive:
            run(binary, "-c", source, "-o", archive)
            return size, "stock"
        return size, "no references"
    run(binary, "-c", source, "-o", archive)
    return 0, "stock"

def restore(archive, output, scratch):
    with archive.open("rb") as data:
        if data.read(len(tag)) == tag:
            native = scratch / "restore-native.bato"
            with native.open("wb") as disk:
                shutil.copyfileobj(data, disk)
            decoded = scratch / "decoded"
            pieces = scratch / "restore-pieces"
            run(binary, "extract", native, "-o", decoded)
            run(helper, "restore", decoded, pieces)
            with output.open("wb") as raw:
                for index, piece in enumerate(sorted(pieces.iterdir())):
                    if not piece.is_file() or piece.name != f"{index:08d}":
                        raise ValueError("invalid restored piece sequence")
                    with piece.open("rb") as disk:
                        shutil.copyfileobj(disk, raw)
            return
    run(binary, "-d", archive, "-o", output)

output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open("w") as output:
    rows = csv.writer(output, delimiter="\t")
    rows.writerow(["input","rep","variant","piece","wrapped","bytes","pack_wall","pack_cpu","restore_wall","restore_cpu","digest"])
    for i, source in enumerate(inputs):
        name = source.name
        for rep in range(args.repetitions):
            shift = (i + rep) % len(variants)
            for variant in variants[shift:] + variants[:shift]:
                with tempfile.TemporaryDirectory(dir=output_path.parent) as tmp:
                    scratch = Path(tmp)
                    archive, restored = (scratch / x for x in ("out.bato", "restored"))
                    start, used = time.monotonic(), cpu()
                    size, outcome = pack(source, archive, scratch, variant)
                    pw, pc = time.monotonic() - start, cpu() - used
                    if outcome in ("single", "no references", "outside prototype limit"):
                        rows.writerow([name,rep,variant,size,outcome,"",pw,pc,"","",""])
                        output.flush()
                        continue
                    start, used = time.monotonic(), cpu()
                    restore(archive, restored, scratch)
                    dw, dc = time.monotonic() - start, cpu() - used
                    subprocess.run(["cmp","-s",source,restored],check=True)
                    data = archive.read_bytes()
                    rows.writerow([name,rep,variant,size,outcome,len(data),pw,pc,dw,dc,hashlib.sha256(data).hexdigest()])
                    output.flush()
        print(f"{name} verified", flush=True)
