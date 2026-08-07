#!/usr/bin/env python3
"""Profile HashEmb bottleneck: per-operation timing breakdown.

Sample model:
    batch_size  = 4096 samples
    feat_ids    = 100 feat_ids per sample  (so total keys = batch × feat_ids)
    feat_space  = size of the feat_id universe (e.g. 1,000,000)

Each batch produces batch × feat_ids int64 keys for lookup.

Run 10 times (configurable) and report avg ± std in milliseconds.

Usage:
    python examples/profile_bottleneck.py
    python examples/profile_bottleneck.py --batch 8192 --feat-ids 200 --feat-space 5000000
"""

import argparse
import statistics
import time

import numpy as np
import torch

from hashemb import HashEmbedding, _hashemb_cpp


def time_repeated(fn, runs=10, warmup=3, prepare=None):
    """Time ``fn()`` ``runs`` times after ``warmup`` calls.

    If *prepare* is given, it is called (untimed) before each invocation
    to reset state — e.g. zero_grad before scatter_add_grad, or
    scatter_add_grad before step().
    """
    for _ in range(warmup):
        if prepare:
            prepare()
        fn()
    times = []
    for _ in range(runs):
        if prepare:
            prepare()
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return statistics.mean(times), statistics.stdev(times)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", type=int, default=4096, help="num samples per batch")
    p.add_argument("--feat-ids", type=int, default=100, help="feat_ids per sample")
    p.add_argument("--feat-space", type=int, default=1_000_000,
                   help="feat_id universe size (keys drawn from [0, feat_space))")
    p.add_argument("--dim", type=int, default=16, help="embedding dim")
    p.add_argument("--capacity", type=int, default=2_000_000)
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--optimizer", type=str, default="adam")
    p.add_argument("--lr", type=float, default=0.001)
    args = p.parse_args()

    B = args.batch          # samples
    F = args.feat_ids       # feat_ids per sample
    D = args.dim
    N = B * F               # total keys per batch
    runs, warmup = args.runs, args.warmup

    # Reproducible random keys: shape (B, F), drawn from [0, feat_space).
    rng = np.random.default_rng(42)
    keys_2d = rng.integers(0, args.feat_space, size=(B, F), dtype=np.int64)
    keys_np = keys_2d.reshape(-1)  # flat (N,) for C++ calls
    unique_keys = np.unique(keys_np)

    print("=" * 64)
    print("HashEmb Bottleneck Profile")
    print("=" * 64)
    print(f"samples={B}  feat_ids/sample={F}  total_keys={N}")
    print(f"feat_space={args.feat_space}  dim={D}  capacity={args.capacity}")
    print(f"optimizer={args.optimizer}  lr={args.lr}")
    print(f"runs={runs}  warmup={warmup}")
    print(f"unique keys in batch: {len(unique_keys)} / {N}")
    print()

    # ══════════════════════════════════════════════════════════════════
    # Part 1: C++ raw operations
    # ══════════════════════════════════════════════════════════════════
    print("-" * 64)
    print("C++ raw operations")
    print("-" * 64)

    table = _hashemb_cpp.HashEmbeddingTable(
        args.capacity, D,
        optimizer=args.optimizer, lr=args.lr,
        block_size=1_000_000,
        initial_scale=0.1,
    )

    # Pre-populate all unique keys so find_or_create is find-only (steady state).
    _, _ = table.lookup_and_gather(unique_keys)
    print(f"  table entries: {table.num_entries}")
    print()

    results_cpp = {}

    # 1. find_or_create — hash table lookup (no insertion in steady state)
    avg, std = time_repeated(lambda: table.find_or_create(keys_np), runs, warmup)
    results_cpp["find_or_create"] = avg
    print(f"  find_or_create      {avg:8.3f} ± {std:.3f} ms")

    # Pre-compute slots for downstream ops
    slots = table.find_or_create(keys_np)

    # 2. lookup — gather embeddings from blocks using pre-computed slots
    avg, std = time_repeated(lambda: table.lookup(slots), runs, warmup)
    results_cpp["lookup(gather)"] = avg
    print(f"  lookup (gather)     {avg:8.3f} ± {std:.3f} ms")

    # 3. lookup_and_gather — find + gather combined (the real forward path)
    avg, std = time_repeated(lambda: table.lookup_and_gather(keys_np), runs, warmup)
    results_cpp["lookup_and_gather"] = avg
    print(f"  lookup_and_gather   {avg:8.3f} ± {std:.3f} ms")

    # 4. lookup_existing — eval path (find_only + gather, no insertion)
    avg, std = time_repeated(lambda: table.lookup_existing(keys_np), runs, warmup)
    results_cpp["lookup_existing"] = avg
    print(f"  lookup_existing     {avg:8.3f} ± {std:.3f} ms")

    # 5. scatter_add_grad — gradient accumulation into grad_blocks_
    grads = rng.standard_normal((N, D)).astype(np.float32)
    avg, std = time_repeated(
        lambda: table.scatter_add_grad(slots, grads),
        runs, warmup,
        prepare=lambda: table.zero_grad(),
    )
    results_cpp["scatter_add_grad"] = avg
    print(f"  scatter_add_grad    {avg:8.3f} ± {std:.3f} ms")

    # 6. step — optimizer update (SGD / Adam) + in-place grad zeroing
    avg, std = time_repeated(
        lambda: table.step(),
        runs, warmup,
        prepare=lambda: table.scatter_add_grad(slots, grads),
    )
    results_cpp["step"] = avg
    print(f"  step                {avg:8.3f} ± {std:.3f} ms")

    # 7. zero_grad — explicit grad clear (for reference)
    avg, std = time_repeated(
        lambda: table.zero_grad(),
        runs, warmup,
        prepare=lambda: table.scatter_add_grad(slots, grads),
    )
    results_cpp["zero_grad"] = avg
    print(f"  zero_grad           {avg:8.3f} ± {std:.3f} ms")

    print()

    # ══════════════════════════════════════════════════════════════════
    # Part 2: PyTorch pipeline (forward / backward / step)
    # ══════════════════════════════════════════════════════════════════
    print("-" * 64)
    print("PyTorch pipeline  (forward → backward → step)")
    print("-" * 64)

    emb = HashEmbedding(
        embedding_dim=D, capacity=args.capacity,
        optimizer=args.optimizer, lr=args.lr,
        initial_scale=0.1,
    )

    # Pre-populate
    _ = emb(torch.from_numpy(unique_keys))
    emb.step()
    print(f"  emb entries: {emb.num_entries}")
    print()

    # keys shape (B, F) → forward output (B, F, D)
    keys_t = torch.from_numpy(keys_2d).to(torch.int64)

    times_fwd, times_bwd, times_step = [], [], []

    # Each cycle: forward → backward → step (real training loop pattern).
    for _ in range(warmup):
        out = emb(keys_t)
        out.sum().backward()
        emb.step()

    for _ in range(runs):
        t0 = time.perf_counter()
        out = emb(keys_t)
        t1 = time.perf_counter()
        out.sum().backward()
        t2 = time.perf_counter()
        emb.step()
        t3 = time.perf_counter()
        times_fwd.append((t1 - t0) * 1000)
        times_bwd.append((t2 - t1) * 1000)
        times_step.append((t3 - t2) * 1000)

    avg_fwd = statistics.mean(times_fwd)
    std_fwd = statistics.stdev(times_fwd)
    avg_bwd = statistics.mean(times_bwd)
    std_bwd = statistics.stdev(times_bwd)
    avg_step = statistics.mean(times_step)
    std_step = statistics.stdev(times_step)

    print(f"  forward             {avg_fwd:8.3f} ± {std_fwd:.3f} ms")
    print(f"  backward            {avg_bwd:8.3f} ± {std_bwd:.3f} ms")
    print(f"  step                {avg_step:8.3f} ± {std_step:.3f} ms")
    total = avg_fwd + avg_bwd + avg_step
    print(f"  {'─' * 40}")
    print(f"  total / cycle       {total:8.3f} ms")
    print()

    # ══════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════
    print("=" * 64)
    print("Summary — C++ operations sorted by cost")
    print("=" * 64)
    total_cpp = sum(results_cpp.values())
    for name, val in sorted(results_cpp.items(), key=lambda x: -x[1]):
        pct = val / total_cpp * 100 if total_cpp > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  {name:20s} {val:8.3f} ms  ({pct:5.1f}%)  {bar}")
    print(f"  {'─' * 40}")
    print(f"  {'total':20s} {total_cpp:8.3f} ms")
    print()


if __name__ == "__main__":
    main()
