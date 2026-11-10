"""Validated HTTP transfers. Completed destinations are never partial files."""
from __future__ import annotations

import http.client
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from .common import atomic_json, sha256


class DownloadError(RuntimeError):
    pass


def _validator(headers):
    tag = headers.get("ETag")
    if tag and not tag.startswith("W/"):
        return tag
    return headers.get("Last-Modified")


def download(urls, destination, *, expected_size=None, expected_sha256=None, timeout=30,
             attempts=3, force=False, progress=None, validate=None, max_bytes=None,
             opener=None, sleep=time.sleep, stop=None):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise DownloadError("Destination must not be a symlink")
    part = destination.with_name(destination.name + ".part")
    metadata_path = part.with_name(part.name + ".json")
    if part.is_symlink() or metadata_path.is_symlink():
        raise DownloadError("Download partials must not be symlinks")
    urls = [urls] if isinstance(urls, str) else list(urls)
    if not urls:
        raise DownloadError("Asset has no URLs")
    if expected_sha256 and destination.is_file() and not force and sha256(destination) == expected_sha256:
        if validate:
            validate(destination)
        return {"sha256": expected_sha256, "size": destination.stat().st_size, "url": urls[0], "reused": True}
    if force:
        part.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
    opener = opener or urllib.request.build_opener()
    errors = []
    for url in urls:
        for attempt in range(attempts):
            if stop and stop.is_set():
                raise DownloadError("Download interrupted")
            state = {}
            if metadata_path.exists():
                try:
                    state = json.loads(metadata_path.read_text())
                except (ValueError, OSError):
                    pass
            offset = part.stat().st_size if part.exists() else 0
            # A partial from another URL/object cannot be safely spliced in.
            if offset and (state.get("url") != url or not state.get("validator") or
                           state.get("expected_sha256") != expected_sha256 or
                           state.get("expected_size") != expected_size):
                part.unlink(missing_ok=True)
                offset = 0
            headers = {"User-Agent": "Bato-Corpus/2", "Accept-Encoding": "identity"}
            if offset:
                headers.update({"Range": f"bytes={offset}-", "If-Range": state["validator"]})
            request = urllib.request.Request(url, headers=headers)
            response = None
            retry_after = None
            try:
                response = opener.open(request, timeout=timeout)
                with response:
                    status = getattr(response, "status", 200)
                    validator = _validator(response.headers)
                    response_length = response.headers.get("Content-Length")
                    response_length = int(response_length) if response_length is not None else None
                    if status == 206:
                        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                        if not match:
                            raise DownloadError("Missing or invalid Content-Range")
                        start, end, total = map(int, match.groups())
                        if start != offset or end < start or end != total - 1:
                            raise DownloadError("Unexpected HTTP range bounds")
                        if response_length is not None and response_length != end - start + 1:
                            raise DownloadError("Content-Length disagrees with range")
                        if offset and validator != state.get("validator"):
                            part.unlink(missing_ok=True)
                            raise DownloadError("Object validator changed during resume")
                        response_length = end - start + 1
                        mode = "ab" if offset else "wb"
                    elif status == 200:
                        offset = 0
                        total = response_length
                        mode = "wb"
                    else:
                        raise DownloadError(f"Unexpected HTTP status {status}")
                    if expected_size is not None and total is not None and total != expected_size:
                        raise DownloadError(f"Source size changed: expected {expected_size}, got {total}")
                    if max_bytes is not None and total is not None and total > max_bytes:
                        raise DownloadError("Download exceeds its transfer budget")
                    atomic_json(metadata_path, {"url": url, "validator": validator, "total": total,
                                               "expected_size": expected_size, "expected_sha256": expected_sha256})
                    transferred = 0
                    started = time.perf_counter()
                    with part.open(mode) as stream:
                        while chunk := response.read(1024 * 1024):
                            if stop and stop.is_set():
                                raise DownloadError("Download interrupted")
                            transferred += len(chunk)
                            if max_bytes is not None and offset + transferred > max_bytes:
                                raise DownloadError("Download exceeds its transfer budget")
                            if response_length is not None and transferred > response_length:
                                raise DownloadError("Response exceeds Content-Length")
                            stream.write(chunk)
                            if progress:
                                progress(offset + transferred, total or expected_size,
                                         transferred / max(time.perf_counter() - started, 1e-9))
                        stream.flush()
                        os.fsync(stream.fileno())
                    if response_length is not None and transferred != response_length:
                        raise DownloadError(f"Truncated response: {transferred}/{response_length} bytes")
                    actual_size = part.stat().st_size
                    known_size = expected_size if expected_size is not None else total
                    if known_size is not None and actual_size != known_size:
                        raise DownloadError(f"Truncated file: {actual_size}/{known_size} bytes")
                    if known_size is None and expected_sha256 is None:
                        raise DownloadError("Source has no length or checksum to verify completion")
                    digest = sha256(part)
                    if expected_sha256 and digest != expected_sha256:
                        part.unlink(missing_ok=True)
                        raise DownloadError("SHA-256 mismatch")
                    if validate:
                        try:
                            validate(part)
                        except Exception:
                            part.unlink(missing_ok=True)
                            raise
                    os.replace(part, destination)
                    metadata_path.unlink(missing_ok=True)
                    return {"sha256": digest, "size": actual_size, "url": url, "reused": False}
            except urllib.error.HTTPError as error:
                error.close()
                errors.append(f"{url}: HTTP {error.code}")
                if error.code == 416:
                    # A complete partial still needs validation. Never accept size alone.
                    if part.exists() and expected_size == part.stat().st_size and expected_sha256 and sha256(part) == expected_sha256:
                        if validate:
                            validate(part)
                        os.replace(part, destination)
                        metadata_path.unlink(missing_ok=True)
                        return {"sha256": expected_sha256, "size": expected_size, "url": url, "reused": False}
                    part.unlink(missing_ok=True)
                elif error.code not in {408, 429, 500, 502, 503, 504}:
                    break
                try:
                    retry_after = min(30, max(0, float(error.headers.get("Retry-After", "0"))))
                except ValueError:
                    pass
            except (OSError, ValueError, DownloadError, EOFError, http.client.HTTPException) as error:
                errors.append(f"{url}: {error}")
            if attempt + 1 < attempts:
                delay = retry_after or min(8, 2 ** attempt + random.random())
                if stop:
                    if stop.wait(delay):
                        raise DownloadError("Download interrupted")
                else:
                    sleep(delay)
    raise DownloadError("; ".join(errors[-6:]) or "All mirrors failed")
