"""Subprocess adapters for standard-library ZIP and BZIP2, explicitly labeled."""
import argparse
import bz2
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bato_tools.common import copy_bounded, regular_files, regular_directories
from bato_tools.corpus import unzip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("codec", choices=["zip", "bzip2"])
    parser.add_argument("action", choices=["compress", "decompress"])
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--directory", action="store_true")
    parser.add_argument("--max-bytes", type=int, default=10_000_000_000)
    parser.add_argument("--level", type=int, choices=range(1, 10), default=9)
    args = parser.parse_args()
    if args.codec == "bzip2":
        if args.action == "compress":
            with args.source.open("rb") as source, bz2.open(args.destination, "wb", compresslevel=args.level) as output:
                copy_bounded(source, output, args.max_bytes)
        else:
            with bz2.open(args.source, "rb") as source, args.destination.open("wb") as output:
                copy_bounded(source, output, args.max_bytes)
    elif args.action == "compress":
        with zipfile.ZipFile(args.destination, "w", zipfile.ZIP_DEFLATED, compresslevel=args.level) as archive:
            for name in regular_directories(args.source) if args.directory else []:
                info = zipfile.ZipInfo(name + "/", (1980, 1, 1, 0, 0, 0))
                info.external_attr = (0o40755 << 16) | 0x10
                archive.writestr(info, b"")
            files = regular_files(args.source) if args.directory else [args.source]
            for file in files:
                name = file.relative_to(args.source).as_posix() if args.directory else "payload"
                info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                if hasattr(info, "compress_level"):
                    info.compress_level = args.level
                else:
                    info._compresslevel = args.level
                info.external_attr = 0o100644 << 16
                info.file_size = file.stat().st_size
                with file.open("rb") as source, archive.open(info, "w", force_zip64=info.file_size > zipfile.ZIP64_LIMIT) as output:
                    copy_bounded(source, output, args.max_bytes)
    elif args.directory:
        args.destination.mkdir()
        unzip(args.source, args.destination, args.max_bytes)
    else:
        with zipfile.ZipFile(args.source) as archive:
            if archive.namelist() != ["payload"]:
                raise ValueError("Unexpected ZIP file inventory")
            with archive.open("payload") as source, args.destination.open("wb") as output:
                copy_bounded(source, output, args.max_bytes)


if __name__ == "__main__":
    main()
