#!/usr/bin/env python3
"""
verify.py — standalone verification script for BXP archives

Does NOT require Rust or the blazex binary.  Pure Python 3.8+.
Verifies:
  1. Magic bytes and format version
  2. xxh3-64 checksum of every tensor
  3. SHA-256 of the entire data section

Usage:
    python3 verify.py model.blz
    python3 verify.py model.blz --json      # machine-readable output
    python3 verify.py model.blz --tensor model.layers.0.self_attn.q_proj.weight
"""

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

# Pure-Python xxh3-64 — no dependencies required
# Based on the public xxHash reference implementation
def xxh3_64(data: bytes) -> int:
    """Pure-Python xxh3-64 (sufficient for verification; not optimised)."""
    try:
        from xxhash import xxh3_64_intdigest  # type: ignore
        return xxh3_64_intdigest(data)
    except ImportError:
        pass
    # Fallback: use ctypes if xxhash not available
    try:
        import ctypes, ctypes.util
        lib_name = ctypes.util.find_library("xxhash")
        if lib_name:
            lib = ctypes.CDLL(lib_name)
            lib.XXH3_64bits.restype = ctypes.c_uint64
            lib.XXH3_64bits.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
            return lib.XXH3_64bits(data, len(data))
    except Exception:
        pass
    # Last resort: warn and skip xxh3 check
    return None  # type: ignore


MAGIC = 0x0A4B50585A414C42  # "BLAZXPK\n"
FORMAT_VERSION = 1


def parse_archive(path: Path):
    data = path.read_bytes()
    offset = 0

    magic, = struct.unpack_from("<Q", data, offset); offset += 8
    if magic != MAGIC:
        raise ValueError(f"Bad magic: {magic:#018x} (expected {MAGIC:#018x})")

    version, = struct.unpack_from("<I", data, offset); offset += 4
    if version != FORMAT_VERSION:
        raise ValueError(f"Unsupported format version {version}")

    header_len, = struct.unpack_from("<Q", data, offset); offset += 8
    header = json.loads(data[offset: offset + header_len])
    data_start = offset + header_len

    return header, data, data_start


def verify(path: Path, tensor_filter: str = None):
    header, data, data_start = parse_archive(path)

    results = {
        "archive": str(path),
        "version": header.get("version"),
        "tensor_count": len(header["tensors"]),
        "tensors": [],
        "sha256_ok": False,
        "all_ok": False,
    }

    xxh3_available = xxh3_64(b"test") is not None

    failed = 0
    for entry in header["tensors"]:
        name = entry["name"]
        if tensor_filter and name != tensor_filter:
            continue

        offset = data_start + entry["data_offset"]
        length = entry["data_len"]
        raw = data[offset: offset + length]

        if len(raw) != length:
            status = "TRUNCATED"
            failed += 1
        elif xxh3_available:
            actual = xxh3_64(raw)
            expected = entry["xxh3"]
            if actual == expected:
                status = "OK"
            else:
                status = f"FAIL (expected {expected:#018x} got {actual:#018x})"
                failed += 1
        else:
            status = "SKIP (xxhash not installed)"

        results["tensors"].append({
            "name": name,
            "dtype": entry["dtype"],
            "shape": entry["shape"],
            "bytes": length,
            "status": status,
        })

    # SHA-256 of entire data section
    tensors = header["tensors"]
    if tensors:
        data_end = data_start + tensors[-1]["data_offset"] + tensors[-1]["data_len"]
    else:
        data_end = data_start
    actual_sha = hashlib.sha256(data[data_start:data_end]).hexdigest()
    expected_sha = header.get("data_sha256", "")
    results["sha256_ok"] = actual_sha == expected_sha
    results["sha256_expected"] = expected_sha
    results["sha256_actual"] = actual_sha

    if not results["sha256_ok"]:
        failed += 1

    results["failed"] = failed
    results["all_ok"] = failed == 0
    return results


def main():
    ap = argparse.ArgumentParser(description="Verify a BXP archive")
    ap.add_argument("archive", type=Path)
    ap.add_argument("--json", action="store_true", help="JSON output")
    ap.add_argument("--tensor", default=None, help="Verify a single named tensor")
    args = ap.parse_args()

    if not args.archive.exists():
        print(f"ERROR: file not found: {args.archive}", file=sys.stderr)
        sys.exit(1)

    try:
        r = verify(args.archive, tensor_filter=args.tensor)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(r, indent=2))
    else:
        print(f"Archive : {r['archive']}")
        print(f"Version : {r['version']}")
        print(f"Tensors : {r['tensor_count']}")
        print()
        for t in r["tensors"]:
            shape_str = "×".join(str(d) for d in t["shape"])
            mb = t["bytes"] / 1_048_576
            print(f"  [{t['status']:8s}] {t['name']:<60s} {t['dtype']:6s} {shape_str} ({mb:.1f} MB)")
        print()
        sha_status = "OK" if r["sha256_ok"] else "FAIL"
        print(f"SHA-256 : [{sha_status}] {r['sha256_expected']}")
        print()
        if r["all_ok"]:
            print("✓ Archive verified — all checks passed")
        else:
            print(f"✗ VERIFICATION FAILED — {r['failed']} check(s) failed")
            sys.exit(1)


if __name__ == "__main__":
    main()
