#!/usr/bin/env python3
"""
Benchmark 10 classic string compression algorithms on 100K random strings.
Measures compression speed, decompression speed, compression ratio, and space savings.
Generated via DeepSeek Prompting.
"""

import bz2
import collections
import gzip
import heapq
import lzma
import os
import pickle
import random
import string
import struct
import sys
import time
import zlib

# ---------------------------------------------------------------------------
# Configuration & Sample Toggle
# ---------------------------------------------------------------------------
NUM_STRINGS = 100_000
MIN_LEN = 5
MAX_LEN = 25
SEED = 42

# All algorithms run on the entire 100,000 strings, so ratios are comparable.
# Set to False to cap the pure-Python algorithms at SAMPLE_SIZE strings.
SAMPLE_ALL = True
SAMPLE_SIZE = 10_000

random.seed(SEED)


# ---------------------------------------------------------------------------
# 1. Dataset Generation
# ---------------------------------------------------------------------------
def generate_random_strings(count=NUM_STRINGS, min_len=MIN_LEN, max_len=MAX_LEN):
    alphabet = string.ascii_letters + string.digits
    return [
        "".join(random.choices(alphabet, k=random.randint(min_len, max_len)))
        for _ in range(count)
    ]


# ---------------------------------------------------------------------------
# 2. Ten Classic Compression Algorithms (Compress & Decompress)
# ---------------------------------------------------------------------------

# 1. Deflate (zlib Level 6 - Standard RFC 1951)
def compress_deflate(data: bytes) -> bytes:
    return zlib.compress(data, level=6)

def decompress_deflate(data: bytes) -> bytes:
    return zlib.decompress(data)


# 2. Fast Deflate (zlib Level 1 - Low Latency Stream)
def compress_fast_deflate(data: bytes) -> bytes:
    return zlib.compress(data, level=1)

def decompress_fast_deflate(data: bytes) -> bytes:
    return zlib.decompress(data)


# 3. Gzip (RFC 1952 Container + Deflate + CRC32)
def compress_gzip(data: bytes) -> bytes:
    return gzip.compress(data, compresslevel=6)

def decompress_gzip(data: bytes) -> bytes:
    return gzip.decompress(data)


# 4. Bzip2 (Burrows-Wheeler Transform + MTF + Huffman)
def compress_bzip2(data: bytes) -> bytes:
    return bz2.compress(data, compresslevel=6)

def decompress_bzip2(data: bytes) -> bytes:
    return bz2.decompress(data)


# 5. LZMA (Lempel-Ziv-Markov Chain / 7-Zip SDK)
def compress_lzma(data: bytes) -> bytes:
    return lzma.compress(data, preset=6)

def decompress_lzma(data: bytes) -> bytes:
    return lzma.decompress(data)


# 6. Run-Length Encoding (RLE)
def compress_rle(data: bytes) -> bytes:
    if not data:
        return b""
    out = bytearray()
    i = 0
    n = len(data)
    while i < n:
        b = data[i]
        run = 1
        while i + 1 < n and data[i + 1] == b and run < 255:
            run += 1
            i += 1
        out.extend([run, b])
        i += 1
    return bytes(out)

def decompress_rle(data: bytes) -> bytes:
    if not data:
        return b""
    out = bytearray()
    for i in range(0, len(data), 2):
        run = data[i]
        b = data[i + 1]
        out.extend([b] * run)
    return bytes(out)


# 7. Huffman Coding (Min-Heap Prefix Code Tree)
def compress_huffman(data: bytes) -> bytes:
    if not data:
        return b""
    counts = collections.Counter(data)
    heap = [[weight, [sym, ""]] for sym, weight in counts.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = "0" + pair[1]
        for pair in hi[1:]:
            pair[1] = "1" + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

    tree = dict(heapq.heappop(heap)[1:]) if heap else {data[0]: "0"}
    tree_header = pickle.dumps(tree)

    encoded_bits = "".join(tree[b] for b in data)
    pad = (8 - len(encoded_bits) % 8) % 8
    encoded_bits += "0" * pad
    byte_vals = bytearray(
        int(encoded_bits[i : i + 8], 2) for i in range(0, len(encoded_bits), 8)
    )

    return struct.pack(">II", len(tree_header), pad) + tree_header + bytes(byte_vals)

def decompress_huffman(data: bytes) -> bytes:
    if not data:
        return b""
    hlen, pad = struct.unpack(">II", data[:8])
    tree = pickle.loads(data[8 : 8 + hlen])
    rev_tree = {v: k for k, v in tree.items()}

    payload = data[8 + hlen :]
    bit_str = "".join(f"{b:08b}" for b in payload)
    if pad:
        bit_str = bit_str[:-pad]

    out = bytearray()
    curr = ""
    for bit in bit_str:
        curr += bit
        if curr in rev_tree:
            out.append(rev_tree[curr])
            curr = ""
    return bytes(out)


# 8. Lempel-Ziv-Welch (LZW 16-Bit Dictionary)
def compress_lzw(data: bytes) -> bytes:
    if not data:
        return b""
    d = {bytes([i]): i for i in range(256)}
    w = bytes()
    res = []
    dict_size = 256
    for b in data:
        wb = w + bytes([b])
        if wb in d:
            w = wb
        else:
            res.append(d[w])
            if dict_size < 65535:
                d[wb] = dict_size
                dict_size += 1
            w = bytes([b])
    if w:
        res.append(d[w])
    return struct.pack(f">{len(res)}H", *res)

def decompress_lzw(data: bytes) -> bytes:
    if not data:
        return b""
    cnt = len(data) // 2
    codes = struct.unpack(f">{cnt}H", data)
    d = {i: bytes([i]) for i in range(256)}
    w = bytes([codes[0]])
    res = bytearray(w)
    dict_size = 256
    for k in codes[1:]:
        if k in d:
            entry = d[k]
        elif k == dict_size:
            entry = w + bytes([w[0]])
        else:
            raise ValueError(f"Corrupt LZW code: {k}")
        res.extend(entry)
        if dict_size < 65535:
            d[dict_size] = w + bytes([entry[0]])
            dict_size += 1
        w = entry
    return bytes(res)


# 9. LZ77 (Sliding Window with Hash Lookahead)
def compress_lz77(data: bytes, window_size=2048) -> bytes:
    if not data:
        return b""
    i = 0
    n = len(data)
    tokens = []
    hash_tbl = {}

    while i < n:
        best_offset = 0
        best_len = 0
        win_start = max(0, i - window_size)

        if i + 3 <= n:
            p3 = data[i : i + 3]
            pos_list = hash_tbl.get(p3, [])
            for p in reversed(pos_list):
                if p < win_start:
                    break
                length = 0
                while i + length < n and data[p + length] == data[i + length] and length < 255:
                    length += 1
                if length > best_len:
                    best_len = length
                    best_offset = i - p
                    if length >= 255:
                        break

        if best_len >= 3:
            next_char = data[i + best_len] if i + best_len < n else 0
            has_next = 1 if i + best_len < n else 0
            tokens.append((best_offset, best_len, has_next, next_char))
            for k in range(best_len + (1 if has_next else 0)):
                if i + k + 3 <= n:
                    hash_tbl.setdefault(data[i + k : i + k + 3], []).append(i + k)
            i += best_len + (1 if has_next else 0)
        else:
            tokens.append((0, 0, 1, data[i]))
            if i + 3 <= n:
                hash_tbl.setdefault(data[i : i + 3], []).append(i)
            i += 1

    out = bytearray(struct.pack(">I", len(tokens)))
    for offset, length, has_next, next_char in tokens:
        out.extend(struct.pack(">HBBB", offset, length, has_next, next_char))
    return bytes(out)

def decompress_lz77(data: bytes) -> bytes:
    if not data:
        return b""
    n_tok = struct.unpack(">I", data[:4])[0]
    out = bytearray()
    pos = 4
    for _ in range(n_tok):
        offset, length, has_next, next_char = struct.unpack(">HBBB", data[pos : pos + 5])
        pos += 5
        if length > 0:
            start = len(out) - offset
            for k in range(length):
                out.append(out[start + k])
        if has_next:
            out.append(next_char)
    return bytes(out)


# 10. Byte-Pair Encoding (BPE Iterative Subword Compression)
def compress_bpe(data: bytes, num_merges=35) -> bytes:
    if not data:
        return b""
    tokens = list(data)
    rules = []
    next_id = 256

    for _ in range(num_merges):
        pairs = {}
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pairs[pair] = pairs.get(pair, 0) + 1
        if not pairs:
            break
        best_pair = max(pairs, key=pairs.get)
        if pairs[best_pair] < 2:
            break

        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                new_tokens.append(next_id)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1

        rules.append((next_id, best_pair[0], best_pair[1]))
        tokens = new_tokens
        next_id += 1

    header = pickle.dumps(rules)
    body = struct.pack(f">{len(tokens)}H", *tokens)
    return struct.pack(">I", len(header)) + header + body

def decompress_bpe(data: bytes) -> bytes:
    if not data:
        return b""
    hlen = struct.unpack(">I", data[:4])[0]
    rules = pickle.loads(data[4 : 4 + hlen])
    body = data[4 + hlen :]
    count = len(body) // 2
    tokens = list(struct.unpack(f">{count}H", body))

    for new_id, b1, b2 in reversed(rules):
        new_tokens = []
        for t in tokens:
            if t == new_id:
                new_tokens.extend([b1, b2])
            else:
                new_tokens.append(t)
        tokens = new_tokens

    return bytes(tokens)


# ---------------------------------------------------------------------------
# 3. Benchmark Suite & Terminal Reporting
# ---------------------------------------------------------------------------
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_log_path = os.path.join(script_dir, "deepseek_compression_results.txt")

    log_lines = []

    def log_print(msg=""):
        print(msg)
        log_lines.append(msg)

    log_print(f"Generating {NUM_STRINGS:,} random strings (5..25 chars)...")
    t0 = time.perf_counter()
    raw_strings = generate_random_strings(NUM_STRINGS)
    gen_time = time.perf_counter() - t0
    log_print(f"Generated {len(raw_strings):,} strings in {gen_time:.4f}s.")

    full_payload = "\n".join(raw_strings).encode("utf-8")
    sample_strings = raw_strings[:SAMPLE_SIZE]
    sample_payload = "\n".join(sample_strings).encode("utf-8")

    if not SAMPLE_ALL:
        log_print(f"NOTE: Pure-Python algorithms run on the first {SAMPLE_SIZE:,} strings (set SAMPLE_ALL=True to override).\n")

    algorithms = [
        ("Deflate (zlib L6)", compress_deflate, decompress_deflate, False),
        ("Fast Deflate (zlib L1)", compress_fast_deflate, decompress_fast_deflate, False),
        ("Gzip (RFC 1952)", compress_gzip, decompress_gzip, False),
        ("Bzip2 (BWT + Huffman)", compress_bzip2, decompress_bzip2, False),
        ("LZMA (7-Zip Core)", compress_lzma, decompress_lzma, False),
        ("Huffman Coding", compress_huffman, decompress_huffman, True),
        ("LZW (Dictionary)", compress_lzw, decompress_lzw, True),
        ("LZ77 (Sliding Window)", compress_lz77, decompress_lz77, True),
        ("Byte-Pair Encoding", compress_bpe, decompress_bpe, True),
        ("Run-Length Encoding", compress_rle, decompress_rle, True),
    ]

    records = []

    for name, comp_fn, decomp_fn, is_pure_python in algorithms:
        target = sample_payload if (is_pure_python and not SAMPLE_ALL) else full_payload
        n_items = len(sample_strings) if (is_pure_python and not SAMPLE_ALL) else NUM_STRINGS

        sys.stdout.write(f"Running {name:<26} on {n_items:>7,} strings ... ")
        sys.stdout.flush()

        t_start = time.perf_counter()
        compressed = comp_fn(target)
        comp_time = time.perf_counter() - t_start

        t_start = time.perf_counter()
        decompressed = decomp_fn(compressed)
        decomp_time = time.perf_counter() - t_start

        total_time = comp_time + decomp_time
        verified = (decompressed == target)
        status_tag = "[OK]" if verified else "[FAIL]"

        orig_sz = len(target)
        comp_sz = len(compressed)
        ratio = (comp_sz / orig_sz) * 100
        savings = (1 - (comp_sz / orig_sz)) * 100
        us_per_item = (total_time / n_items) * 1_000_000
        mb_per_sec = (orig_sz / 1024 / 1024) / total_time if total_time > 0 else 0

        sys.stdout.write(f"{total_time:.4f}s   {status_tag}\n")
        log_lines.append(f"Running {name:<26} on {n_items:>7,} strings ... {total_time:.4f}s   {status_tag}")

        records.append({
            "name": name,
            "total_time": total_time,
            "comp_time": comp_time,
            "decomp_time": decomp_time,
            "n": n_items,
            "us_per_item": us_per_item,
            "mb_per_sec": mb_per_sec,
            "orig_sz": orig_sz,
            "comp_sz": comp_sz,
            "ratio": ratio,
            "savings": savings,
            "verified": verified
        })

    # Sort by total execution time (Fastest to Slowest)
    ranked_speed = sorted(records, key=lambda x: x["total_time"])

    log_print("\n" + "=" * 90)
    log_print("TIMING & PERFORMANCE RESULTS — FASTEST TO SLOWEST")
    log_print("=" * 90)
    log_print(f"{'Rank':<5} {'Algorithm':<26} {'Total (s)':<11} {'Comp (s)':<10} {'Decomp (s)':<11} {'N':<9} {'µs/item':<9} {'MB/s':<8}")
    log_print("-" * 90)

    for rank, r in enumerate(ranked_speed, 1):
        log_print(
            f"{rank:<5} {r['name']:<26} {r['total_time']:10.4f} {r['comp_time']:9.4f} {r['decomp_time']:10.4f} "
            f"{r['n']:>8,} {r['us_per_item']:8.2f} {r['mb_per_sec']:7.2f}"
        )
    log_print("=" * 90)

    # Sort by compression ratio (Best to Worst)
    ranked_ratio = sorted(records, key=lambda x: x["ratio"])

    log_print("\n" + "=" * 90)
    log_print("COMPRESSION RATIO & SPACE SAVINGS RANKING (BEST TO WORST)")
    log_print("=" * 90)
    log_print(f"{'Rank':<5} {'Algorithm':<26} {'Orig (KB)':<11} {'Comp (KB)':<11} {'Ratio (%)':<11} {'Savings (%)':<12} {'Verified'}")
    log_print("-" * 90)

    for rank, r in enumerate(ranked_ratio, 1):
        verified_str = "PASSED" if r["verified"] else "FAILED"
        log_print(
            f"{rank:<5} {r['name']:<26} {r['orig_sz']/1024:10.2f} {r['comp_sz']/1024:10.2f} "
            f"{r['ratio']:10.2f}% {r['savings']:11.2f}% {verified_str:>8}"
        )
    log_print("=" * 90)

    log_print(f"\n🏆 Fastest Speed:     {ranked_speed[0]['name']} ({ranked_speed[0]['total_time']:.4f}s)")
    log_print(f"🐢 Slowest Speed:     {ranked_speed[-1]['name']} ({ranked_speed[-1]['total_time']:.4f}s)")
    log_print(f"🥇 Best Compression:  {ranked_ratio[0]['name']} ({ranked_ratio[0]['ratio']:.2f}% size / {ranked_ratio[0]['savings']:.2f}% savings)")
    log_print(f"❌ Worst Compression: {ranked_ratio[-1]['name']} ({ranked_ratio[-1]['ratio']:.2f}% size)")

    with open(output_log_path, "w") as f:
        f.write("\n".join(log_lines) + "\n")

    print(f"\nDetailed benchmark log written to: {output_log_path}")


if __name__ == "__main__":
    main()
