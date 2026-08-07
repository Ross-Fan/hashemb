#!/usr/bin/env python3
"""Profile HashEmb training pipeline: forward → backward → step.

Simulates a real training loop with a dense model on accelerator (CUDA/MPS/CPU),
and breaks down each stage into C++ vs PyTorch/transfer overhead.

Usage:
    python examples/profile_training.py
    python examples/profile_training.py --batch-size 4096 --feats 233 --dim 16 --n-iter 20
"""

import argparse
import statistics
import time

import numpy as np
import torch
import torch.nn as nn

from hashemb import HashEmbedding
from hashemb.utils import get_device


def sync_device(device):
    """Synchronize accelerator for accurate wall-clock timing."""
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        if hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-size", type=int, default=4096, help="samples per batch")
    p.add_argument("--feats", type=int, default=233, help="feat_ids per sample")
    p.add_argument("--dim", type=int, default=16, help="embedding dim")
    p.add_argument("--capacity", type=int, default=1_000_000, help="table capacity")
    p.add_argument("--n-iter", type=int, default=20, help="timed iterations")
    p.add_argument("--warmup", type=int, default=5, help="warmup iterations")
    p.add_argument("--optimizer", type=str, default="adam")
    p.add_argument("--lr", type=float, default=0.001)
    args = p.parse_args()

    B, F, D = args.batch_size, args.feats, args.dim
    N = B * F  # total keys per batch
    device = get_device()

    # ── Header ─────────────────────────────────────────────────────
    print(f"  Device: {device}")
    print(f"  Batch: {B:,} × {F} feats = {N:,} keys")
    print(f"  Dim: {D},  Table: {args.capacity:,} entries")

    # ── Create models ──────────────────────────────────────────────
    emb = HashEmbedding(
        embedding_dim=D, capacity=args.capacity,
        optimizer=args.optimizer, lr=args.lr,
        initial_scale=0.1,
    )
    dense = nn.Linear(D, 1).to(device)
    optimizer = torch.optim.SGD(dense.parameters(), lr=0.01)

    # Pre-populate table so forward is find-only (steady state).
    populate_n = min(args.capacity, max(N * 2, 100_000))
    all_keys = torch.arange(populate_n, dtype=torch.int64)
    _ = emb(all_keys)
    emb.step()
    print(f"  Entries: {emb.num_entries:,}")
    print()

    # Generate fixed keys for all iterations (reproducible).
    rng = np.random.default_rng(42)
    n_total = args.warmup + args.n_iter
    keys_batch = [
        rng.integers(0, populate_n, size=(B, F), dtype=np.int64)
        for _ in range(n_total)
    ]

    # ══════════════════════════════════════════════════════════════════
    # Part 1: Full training loop — measure forward / backward / step
    # ══════════════════════════════════════════════════════════════════
    fwd_times, bwd_times, step_times = [], [], []

    for i in range(n_total):
        keys_t = torch.from_numpy(keys_batch[i]).to(device)

        # ── Forward ───────────────────────────────────────────────
        sync_device(device)
        t0 = time.perf_counter()
        emb_out = emb(keys_t)            # (B, F, D) on device
        dense_out = dense(emb_out)       # (B, F, 1) on device
        loss = dense_out.sum()
        sync_device(device)
        t1 = time.perf_counter()

        # ── Backward ──────────────────────────────────────────────
        loss.backward()
        sync_device(device)
        t2 = time.perf_counter()

        # ── Step ──────────────────────────────────────────────────
        optimizer.step()
        emb.step()
        sync_device(device)
        t3 = time.perf_counter()

        if i >= args.warmup:
            fwd_times.append((t1 - t0) * 1000)
            bwd_times.append((t2 - t1) * 1000)
            step_times.append((t3 - t2) * 1000)

    fwd_avg = statistics.mean(fwd_times)
    bwd_avg = statistics.mean(bwd_times)
    step_avg = statistics.mean(step_times)
    total_avg = fwd_avg + bwd_avg + step_avg

    # ══════════════════════════════════════════════════════════════════
    # Part 2: C++ raw operations (no autograd, no transfer)
    # ══════════════════════════════════════════════════════════════════
    table = emb._table

    # C++ lookup_and_gather (forward C++ part)
    c_fwd_times = []
    slots = None
    for i in range(args.n_iter):
        k = keys_batch[args.warmup + i].reshape(-1)
        t0 = time.perf_counter()
        _, slots = table.lookup_and_gather(k)
        t1 = time.perf_counter()
        c_fwd_times.append((t1 - t0) * 1000)
    c_fwd_avg = statistics.mean(c_fwd_times)

    # C++ scatter_add_grad (backward C++ part)
    grads_np = rng.standard_normal((N, D)).astype(np.float32)
    c_scatter_times = []
    for i in range(args.n_iter):
        table.zero_grad()
        t0 = time.perf_counter()
        table.scatter_add_grad(slots, grads_np)
        t1 = time.perf_counter()
        c_scatter_times.append((t1 - t0) * 1000)
    c_scatter_avg = statistics.mean(c_scatter_times)

    # C++ step (optimizer update + fused in-place zero)
    c_step_times = []
    for i in range(args.n_iter):
        table.scatter_add_grad(slots, grads_np)
        t0 = time.perf_counter()
        table.step()
        t1 = time.perf_counter()
        c_step_times.append((t1 - t0) * 1000)
    c_step_avg = statistics.mean(c_step_times)

    # C++ zero_grad (standalone, for reference)
    c_zg_times = []
    for i in range(args.n_iter):
        table.scatter_add_grad(slots, grads_np)
        t0 = time.perf_counter()
        table.zero_grad()
        t1 = time.perf_counter()
        c_zg_times.append((t1 - t0) * 1000)
    c_zg_avg = statistics.mean(c_zg_times)

    # ══════════════════════════════════════════════════════════════════
    # Part 3: Compute breakdowns
    # ══════════════════════════════════════════════════════════════════
    # Forward overhead = D2H keys + H2D embeddings + dense forward + autograd graph
    py_fwd = fwd_avg - c_fwd_avg
    # Backward overhead = GPU dense backward + D2H gradients
    gpu_bwd = bwd_avg - c_scatter_avg
    # Step overhead = dense optimizer.step()
    dense_step = step_avg - c_step_avg

    # ══════════════════════════════════════════════════════════════════
    # Part 4: Print results
    # ══════════════════════════════════════════════════════════════════
    print("  --- C++ layer profiling (no autograd overhead) ---")
    print()
    print("  =================================================================")
    print(f"    Breakdown (avg over {args.n_iter} iterations)")
    print("  =================================================================")
    print(f"    {'Stage':<36} {'Time(ms)':>8}  {'%':>5}")
    print(f"    {'─' * 36} {'─' * 8} ─{'─' * 5}")

    # Forward
    print(f"    {'Forward total':<36} {fwd_avg:>8.2f}  {fwd_avg / total_avg * 100:>5.1f}")
    print(f"    {'  C++ lookup_and_gather':<36} {c_fwd_avg:>8.2f}  {c_fwd_avg / total_avg * 100:>5.1f}")
    print(f"    {'  PyTorch + transfer overhead':<36} {py_fwd:>8.2f}  {py_fwd / total_avg * 100:>5.1f}")

    # Backward
    print(f"    {'Backward total':<36} {bwd_avg:>8.2f}  {bwd_avg / total_avg * 100:>5.1f}")
    print(f"    {'  GPU dense grad + D2H transfer':<36} {gpu_bwd:>8.2f}  {gpu_bwd / total_avg * 100:>5.1f}")
    print(f"    {'  C++ scatter_add_grad':<36} {c_scatter_avg:>8.2f}  {c_scatter_avg / total_avg * 100:>5.1f}")

    # Step
    print(f"    {'Step total':<36} {step_avg:>8.2f}  {step_avg / total_avg * 100:>5.1f}")
    print(f"    {'  C++ emb.step() (Adam + fused zero)':<36} {c_step_avg:>8.2f}  {c_step_avg / total_avg * 100:>5.1f}")
    print(f"    {'  Dense optimizer.step()':<36} {dense_step:>8.2f}  {dense_step / total_avg * 100:>5.1f}")

    print(f"    {'─' * 36} {'─' * 8} ─{'─' * 5}")
    print(f"    {'TOTAL':<36} {total_avg:>8.2f}  {100.0:>5.1f}")
    print("  =================================================================")
    print()

    # Metrics
    keys_per_sec = N / (total_avg / 1000)
    fwd_bwd_ratio = fwd_avg / bwd_avg if bwd_avg > 0 else float("inf")
    print(f"    Keys/sec: {keys_per_sec / 1e6:.1f}M")
    if fwd_avg > bwd_avg:
        print(f"    Forward/Backward ratio: {fwd_bwd_ratio:.2f}x (fwd > bwd)")
    else:
        print(f"    Forward/Backward ratio: {1 / fwd_bwd_ratio:.2f}x (bwd > fwd)")
    print()

    # Bottleneck ranking
    print("    Bottleneck ranking:")
    items = [
        ("C++ lookup_and_gather (fwd)", c_fwd_avg),
        ("C++ scatter_add_grad (bwd)", c_scatter_avg),
        ("GPU dense grad + transfer (bwd)", gpu_bwd),
        ("PyTorch + transfer overhead (fwd)", py_fwd),
        ("C++ emb.step() (step)", c_step_avg),
        ("Dense optimizer.step() (step)", dense_step),
    ]
    items.sort(key=lambda x: -x[1])
    for rank, (name, val) in enumerate(items, 1):
        pct = val / total_avg * 100
        bar = "█" * max(1, int(pct / 2))
        print(f"    {rank}. {name:<40} {pct:5.1f}% {bar}")
    print()


if __name__ == "__main__":
    main()
