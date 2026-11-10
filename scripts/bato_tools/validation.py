"""Lightweight structural checks; provenance and hashes are tracked separately."""
import struct
import wave
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path


def validate_file(path, kind="auto"):
    path = Path(path)
    kind = path.suffix.lower().lstrip(".") if kind == "auto" else kind
    size = path.stat().st_size
    with path.open("rb") as stream:
        prefix = stream.read(64)
    if kind in {"docx", "xlsx", "pptx", "odt", "ods", "odp", "apk", "aab", "zip"}:
        required = {"docx": "word/document.xml", "xlsx": "xl/workbook.xml", "pptx": "ppt/presentation.xml"}
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None:
                raise ValueError("ZIP member CRC failed")
            if kind in required:
                for name in ["[Content_Types].xml", "_rels/.rels", required[kind]]:
                    if name not in archive.namelist():
                        raise ValueError(f"Missing Office part: {name}")
                for member in archive.infolist():
                    if member.filename.endswith((".xml", ".rels")):
                        if member.file_size > 32 * 1024 * 1024:
                            raise ValueError("Oversized Office XML part")
                        ET.fromstring(archive.read(member))
    elif kind == "png":
        import zlib
        with path.open("rb") as stream:
            if stream.read(8) != b"\x89PNG\r\n\x1a\n":
                raise ValueError("Invalid PNG signature")
            seen_header = seen_data = False
            while True:
                header = stream.read(8)
                if len(header) != 8:
                    raise ValueError("Truncated PNG")
                length, chunk_type = struct.unpack(">I4s", header)
                if length > size:
                    raise ValueError("Invalid PNG chunk length")
                crc = zlib.crc32(chunk_type)
                remaining = length
                while remaining:
                    payload = stream.read(min(remaining, 1024 * 1024))
                    if not payload:
                        raise ValueError("Truncated PNG chunk")
                    remaining -= len(payload)
                    crc = zlib.crc32(payload, crc)
                checksum = stream.read(4)
                if len(checksum) != 4 or struct.unpack(">I", checksum)[0] != crc & 0xffffffff:
                    raise ValueError("PNG CRC mismatch")
                seen_header |= chunk_type == b"IHDR"
                seen_data |= chunk_type == b"IDAT"
                if chunk_type == b"IEND":
                    if not seen_header or not seen_data or length or stream.read(1):
                        raise ValueError("Invalid PNG structure")
                    break
    elif kind in {"jpg", "jpeg"}:
        with path.open("rb") as stream:
            stream.seek(-2, 2)
            ending = stream.read(2)
        if not prefix.startswith(b"\xff\xd8") or ending != b"\xff\xd9":
            raise ValueError("Invalid JPEG markers")
    elif kind == "wav":
        with wave.open(str(path), "rb") as stream:
            expected = stream.getnframes() * stream.getnchannels() * stream.getsampwidth()
            actual = 0
            while block := stream.readframes(65536):
                actual += len(block)
            if actual != expected:
                raise ValueError("Truncated WAV audio")
    elif kind == "flac" and not prefix.startswith(b"fLaC"):
        raise ValueError("Invalid FLAC signature")
    elif kind == "mp4":
        validate_mp4(path)
    elif kind == "iso":
        with path.open("rb") as stream:
            stream.seek(16 * 2048 + 1)
            if stream.read(5) != b"CD001":
                raise ValueError("Invalid ISO volume descriptor")
    elif kind in {"exe", "dll"} and not prefix.startswith(b"MZ"):
        raise ValueError("Invalid PE executable signature")
    elif kind == "pdf" and not prefix.startswith(b"%PDF-"):
        raise ValueError("Invalid PDF signature")
    elif kind == "json":
        import json
        if size > 64 * 1024 * 1024:
            raise ValueError("JSON exceeds validation memory budget")
        json.loads(path.read_text(encoding="utf-8"))
    return {"validation": "passed", "validation_kind": kind, "validation_scope": "structural"}


def validate_mp4(path):
    size = path.stat().st_size
    found_type = found_data = found_movie = False
    with path.open("rb") as stream:
        offset = 0
        while offset < size:
            stream.seek(offset)
            header = stream.read(8)
            if len(header) != 8:
                raise ValueError("Truncated MP4 box")
            length, kind = struct.unpack(">I4s", header)
            header_size = 8
            if length == 1:
                extra = stream.read(8)
                if len(extra) != 8:
                    raise ValueError("Truncated MP4 extended size")
                length = struct.unpack(">Q", extra)[0]
                header_size = 16
            elif length == 0:
                length = size - offset
            if length < header_size or offset + length > size:
                raise ValueError("MP4 box extends beyond file")
            found_type |= kind == b"ftyp"
            found_data |= kind == b"mdat"
            found_movie |= kind == b"moov"
            if kind == b"moov":
                if length > 32 * 1024 * 1024:
                    raise ValueError("MP4 metadata exceeds validation memory budget")
                _validate_movie(stream.read(length - header_size), size)
            offset += length
    if not (found_type and found_data and found_movie):
        raise ValueError("MP4 lacks ftyp, moov, or mdat")


def _validate_movie(data, file_size):
    offset = 0
    while offset < len(data):
        if offset + 8 > len(data):
            raise ValueError("Truncated MP4 metadata")
        length, kind = struct.unpack_from(">I4s", data, offset)
        if length < 8 or offset + length > len(data):
            raise ValueError("Invalid MP4 metadata box")
        payload = data[offset + 8:offset + length]
        if kind in {b"trak", b"mdia", b"minf", b"stbl"}:
            _validate_movie(payload, file_size)
        elif kind in {b"stco", b"co64"}:
            if len(payload) < 8:
                raise ValueError("Truncated MP4 chunk table")
            count = struct.unpack_from(">I", payload, 4)[0]
            width = 4 if kind == b"stco" else 8
            if 8 + count * width > len(payload):
                raise ValueError("Truncated MP4 chunk offsets")
            for index in range(count):
                chunk = int.from_bytes(payload[8 + index * width:8 + (index + 1) * width], "big")
                if chunk >= file_size:
                    raise ValueError("MP4 sample references bytes beyond the file")
        offset += length
