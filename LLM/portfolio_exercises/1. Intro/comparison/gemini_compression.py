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
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt
import pandas as pd

# ==========================================
# Configuration
# ==========================================
NUM_STRINGS = 100_000
MIN_LEN = 5
MAX_LEN = 25
SEED = 42

random.seed(SEED)


# ==========================================
# 1. Data Generation
# ==========================================
def generate_random_strings(count=NUM_STRINGS, min_l=MIN_LEN, max_l=MAX_LEN):
    """Generate a list of random strings of length between min_l and max_l."""
    alphabet = string.ascii_letters + string.digits
    return [
        "".join(random.choices(alphabet, k=random.randint(min_l, max_l)))
        for _ in range(count)
    ]


# ==========================================
# 2. Ten Classic Compression Algorithms
# ==========================================

# 1. Deflate (zlib Level 6 - Standard)
def deflate_compress(data: bytes) -> bytes:
    return zlib.compress(data, level=6)

def deflate_decompress(data: bytes) -> bytes:
    return zlib.decompress(data)


# 2. Fast Deflate (zlib Level 1 - Real-time)
def fast_deflate_compress(data: bytes) -> bytes:
    return zlib.compress(data, level=1)

def fast_deflate_decompress(data: bytes) -> bytes:
    return zlib.decompress(data)


# 3. Gzip (RFC 1952 Framing + Deflate)
def gzip_compress(data: bytes) -> bytes:
    return gzip.compress(data, compresslevel=6)

def gzip_decompress(data: bytes) -> bytes:
    return gzip.decompress(data)


# 4. Bzip2 (Burrows-Wheeler Transform + Huffman)
def bzip2_compress(data: bytes) -> bytes:
    return bz2.compress(data, compresslevel=6)

def bzip2_decompress(data: bytes) -> bytes:
    return bz2.decompress(data)


# 5. LZMA (Lempel-Ziv-Markov Chain / 7-Zip)
def lzma_compress(data: bytes) -> bytes:
    return lzma.compress(data, preset=6)

def lzma_decompress(data: bytes) -> bytes:
    return lzma.decompress(data)


# 6. Run-Length Encoding (RLE)
def rle_compress(data: bytes) -> bytes:
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

def rle_decompress(data: bytes) -> bytes:
    if not data:
        return b""
    out = bytearray()
    for i in range(0, len(data), 2):
        run = data[i]
        b = data[i + 1]
        out.extend([b] * run)
    return bytes(out)


# 7. Huffman Coding (Min-Heap Prefix Tree)
def huffman_compress(data: bytes) -> bytes:
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
    tree_bytes = pickle.dumps(tree)

    encoded_bits = "".join(tree[b] for b in data)
    pad = (8 - len(encoded_bits) % 8) % 8
    encoded_bits += "0" * pad
    byte_vals = bytearray(
        int(encoded_bits[i : i + 8], 2) for i in range(0, len(encoded_bits), 8)
    )

    header = struct.pack(">II", len(tree_bytes), pad)
    return header + tree_bytes + bytes(byte_vals)

def huffman_decompress(data: bytes) -> bytes:
    if not data:
        return b""
    tree_len, pad = struct.unpack(">II", data[:8])
    tree = pickle.loads(data[8 : 8 + tree_len])
    reverse_tree = {v: k for k, v in tree.items()}

    payload = data[8 + tree_len :]
    bit_str = "".join(f"{b:08b}" for b in payload)
    if pad > 0:
        bit_str = bit_str[:-pad]

    out = bytearray()
    curr = ""
    for bit in bit_str:
        curr += bit
        if curr in reverse_tree:
            out.append(reverse_tree[curr])
            curr = ""
    return bytes(out)


# 8. Lempel-Ziv-Welch (LZW Dictionary Encoding)
def lzw_compress(data: bytes) -> bytes:
    if not data:
        return b""
    d_map = {bytes([i]): i for i in range(256)}
    w = bytes()
    result = []
    dict_size = 256
    for b in data:
        wb = w + bytes([b])
        if wb in d_map:
            w = wb
        else:
            result.append(d_map[w])
            if dict_size < 65535:
                d_map[wb] = dict_size
                dict_size += 1
            w = bytes([b])
    if w:
        result.append(d_map[w])
    return struct.pack(f">{len(result)}H", *result)

def lzw_decompress(data: bytes) -> bytes:
    if not data:
        return b""
    count = len(data) // 2
    codes = struct.unpack(f">{count}H", data)
    d_map = {i: bytes([i]) for i in range(256)}
    w = bytes([codes[0]])
    result = bytearray(w)
    dict_size = 256
    for k in codes[1:]:
        if k in d_map:
            entry = d_map[k]
        elif k == dict_size:
            entry = w + bytes([w[0]])
        else:
            raise ValueError(f"Invalid LZW code: {k}")
        result.extend(entry)
        if dict_size < 65535:
            d_map[dict_size] = w + bytes([entry[0]])
            dict_size += 1
        w = entry
    return bytes(result)


# 9. LZ77 (Sliding Window with Hash-Table Lookahead)
def lz77_compress(data: bytes, window_size=2048) -> bytes:
    if not data:
        return b""
    i = 0
    n = len(data)
    tokens = []
    hash_table = {}

    while i < n:
        best_offset = 0
        best_length = 0
        window_start = max(0, i - window_size)

        if i + 3 <= n:
            prefix = data[i : i + 3]
            pos_list = hash_table.get(prefix, [])
            for p in reversed(pos_list):
                if p < window_start:
                    break
                length = 0
                while i + length < n and data[p + length] == data[i + length] and length < 255:
                    length += 1
                if length > best_length:
                    best_length = length
                    best_offset = i - p
                    if length >= 255:
                        break

        if best_length >= 3:
            next_char = data[i + best_length] if i + best_length < n else 0
            has_next = 1 if i + best_length < n else 0
            tokens.append((best_offset, best_length, has_next, next_char))
            for k in range(best_length + (1 if has_next else 0)):
                if i + k + 3 <= n:
                    p3 = data[i + k : i + k + 3]
                    hash_table.setdefault(p3, []).append(i + k)
            i += best_length + (1 if has_next else 0)
        else:
            tokens.append((0, 0, 1, data[i]))
            if i + 3 <= n:
                hash_table.setdefault(data[i : i + 3], []).append(i)
            i += 1

    out = bytearray(struct.pack(">I", len(tokens)))
    for offset, length, has_next, next_char in tokens:
        out.extend(struct.pack(">HBBB", offset, length, has_next, next_char))
    return bytes(out)

def lz77_decompress(data: bytes) -> bytes:
    if not data:
        return b""
    num_tokens = struct.unpack(">I", data[:4])[0]
    out = bytearray()
    pos = 4
    for _ in range(num_tokens):
        offset, length, has_next, next_char = struct.unpack(">HBBB", data[pos : pos + 5])
        pos += 5
        if length > 0:
            start = len(out) - offset
            for k in range(length):
                out.append(out[start + k])
        if has_next:
            out.append(next_char)
    return bytes(out)


# 10. Byte-Pair Encoding (BPE - Iterative Subword Substitution)
def bpe_compress(data: bytes, num_merges=35) -> bytes:
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

def bpe_decompress(data: bytes) -> bytes:
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


# ==========================================
# 3. Main Benchmark Execution & Verification
# ==========================================
def main():
    print("=" * 75)
    print(" 10 CLASSIC STRING COMPRESSION ALGORITHMS BENCHMARK (GEMINI)")
    print("=" * 75)

    print(f"Generating {NUM_STRINGS:,} random strings (length: {MIN_LEN}..{MAX_LEN})...")
    start_gen = time.perf_counter()
    strings_list = generate_random_strings(NUM_STRINGS, MIN_LEN, MAX_LEN)
    gen_time = time.perf_counter() - start_gen
    print(f"Generated {len(strings_list):,} strings in {gen_time:.3f} seconds.")

    # Serialize into canonical byte payload (newline-separated)
    raw_bytes = "\n".join(strings_list).encode("utf-8")
    raw_size = len(raw_bytes)
    print(f"Raw data payload size: {raw_size:,} bytes ({raw_size / 1024 / 1024:.2f} MB)\n")

    # Every algorithm runs on the full 100K payload so the ratios are comparable.
    # Lower this to cap the pure-Python algorithms at a smaller sample.
    SAMPLE_STRINGS_COUNT = NUM_STRINGS
    sample_bytes = "\n".join(strings_list[:SAMPLE_STRINGS_COUNT]).encode("utf-8")
    sample_size = len(sample_bytes)

    algorithms = [
        ("Deflate (zlib L6)", deflate_compress, deflate_decompress, False),
        ("Fast Deflate (zlib L1)", fast_deflate_compress, fast_deflate_decompress, False),
        ("Gzip (RFC 1952)", gzip_compress, gzip_decompress, False),
        ("Bzip2 (BWT + Huffman)", bzip2_compress, bzip2_decompress, False),
        ("LZMA (7-Zip Core)", lzma_compress, lzma_decompress, False),
        ("Huffman Coding (Entropy)", huffman_compress, huffman_decompress, True),
        ("LZW (Dictionary)", lzw_compress, lzw_decompress, True),
        ("LZ77 (Sliding Window)", lz77_compress, lz77_decompress, True),
        ("Byte-Pair Encoding (BPE)", bpe_compress, bpe_decompress, True),
        ("Run-Length Encoding (RLE)", rle_compress, rle_decompress, True),
    ]

    results = []

    print("=" * 75)
    print(f"{'Algorithm':<26} | {'Comp (s)':<9} | {'Decomp (s)':<10} | {'Ratio':<8} | {'Status'}")
    print("-" * 75)

    for name, comp_fn, decomp_fn, is_sample in algorithms:
        target_payload = sample_bytes if is_sample else raw_bytes
        orig_len = len(target_payload)

        # 1. Measure compression
        t0 = time.perf_counter()
        compressed = comp_fn(target_payload)
        t_comp = time.perf_counter() - t0

        # 2. Measure decompression
        t0 = time.perf_counter()
        decompressed = decomp_fn(compressed)
        t_decomp = time.perf_counter() - t0

        # 3. Verify integrity
        is_identical = decompressed == target_payload
        status = "PASSED" if is_identical else "FAILED"

        comp_len = len(compressed)
        ratio = (comp_len / orig_len) * 100
        savings = (1 - (comp_len / orig_len)) * 100
        total_time = t_comp + t_decomp

        sample_note = f" (N={SAMPLE_STRINGS_COUNT:,})" if is_sample else f" (N={NUM_STRINGS:,})"

        results.append({
            "Algorithm": name + sample_note,
            "Algorithm Name": name,
            "Comp Time (s)": round(t_comp, 4),
            "Decomp Time (s)": round(t_decomp, 4),
            "Total Time (s)": round(total_time, 4),
            "Orig Size (KB)": round(orig_len / 1024, 2),
            "Comp Size (KB)": round(comp_len / 1024, 2),
            "Ratio (%)": round(ratio, 2),
            "Savings (%)": round(savings, 2),
            "Verified": status,
        })

        print(f"{name:<26} | {t_comp:8.4f}s | {t_decomp:9.4f}s | {ratio:6.2f}% | [{status}]")

    df = pd.DataFrame(results)

    # 1. Rank by Speed (Fastest to Slowest Total Time)
    df_speed = df.sort_values(by="Total Time (s)", ascending=True).reset_index(drop=True)
    df_speed.index += 1
    df_speed.index.name = "Rank (Speed)"

    print("\n" + "=" * 80)
    print("RANKING BY EXECUTION SPEED (FASTEST TO SLOWEST)")
    print("=" * 80)
    print(df_speed[["Algorithm", "Comp Time (s)", "Decomp Time (s)", "Total Time (s)", "Verified"]].to_string())

    # 2. Rank by Compression Ratio (Lowest Ratio = Highest Compression)
    df_ratio = df.sort_values(by="Ratio (%)", ascending=True).reset_index(drop=True)
    df_ratio.index += 1
    df_ratio.index.name = "Rank (Ratio)"

    print("\n" + "=" * 80)
    print("RANKING BY COMPRESSION RATIO (BEST TO WORST)")
    print("=" * 80)
    print(df_ratio[["Algorithm", "Orig Size (KB)", "Comp Size (KB)", "Ratio (%)", "Savings (%)"]].to_string())
    print("=" * 80)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Save visual comparison chart
    output_png = os.path.join(script_dir, "compression_benchmark_results.png")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Subplot 1: Speed (Stacked bar: Comp + Decomp Time)
    algorithms_plot = df_speed["Algorithm Name"][::-1]
    comp_times = df_speed["Comp Time (s)"][::-1]
    decomp_times = df_speed["Decomp Time (s)"][::-1]

    ax1.barh(algorithms_plot, comp_times, label="Compression Time", color="#2563eb")
    ax1.barh(algorithms_plot, decomp_times, left=comp_times, label="Decompression Time", color="#38bdf8")
    ax1.set_xlabel("Time (seconds)", fontsize=11, fontweight="bold")
    ax1.set_title("Execution Speed (Comp vs Decomp)", fontsize=13, fontweight="bold")
    ax1.grid(axis="x", linestyle="--", alpha=0.6)
    ax1.legend(loc="center right")

    # Subplot 2: Compression Ratio (% of original size, lower is better)
    df_ratio_sorted = df.sort_values(by="Ratio (%)", ascending=False)
    colors = ["#16a34a" if r < 100 else "#dc2626" for r in df_ratio_sorted["Ratio (%)"]]
    bars2 = ax2.barh(df_ratio_sorted["Algorithm Name"], df_ratio_sorted["Ratio (%)"], color=colors)
    ax2.axvline(100, color="#6b7280", linestyle="--", linewidth=1.2, label="Uncompressed Baseline (100%)")
    ax2.set_xlabel("Compressed Size (% of Original)", fontsize=11, fontweight="bold")
    ax2.set_title("Compression Ratio (Lower is Better)", fontsize=13, fontweight="bold")
    ax2.grid(axis="x", linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right")

    for bar in bars2:
        val = bar.get_width()
        ax2.text(
            val + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%",
            va="center",
            fontsize=9,
            fontweight="bold"
        )

    plt.suptitle("Benchmark: 10 Classic String Compression Algorithms", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    print(f"\n Visual benchmark chart saved to: {os.path.abspath(output_png)}")

    # Save text report
    output_txt = os.path.join(script_dir, "gemini_compression_results.txt")
    with open(output_txt, "w") as f:
        f.write("GEMINI STRING COMPRESSION BENCHMARK REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(df_speed.to_string() + "\n\n")
        f.write(df_ratio.to_string() + "\n")
    print(f" Summary text results saved to: {os.path.abspath(output_txt)}")


if __name__ == "__main__":
    main()
