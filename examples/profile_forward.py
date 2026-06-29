#!/usr/bin/env python3
"""
Profile the forward pass to identify bottlenecks.
"""
import sys
sys.path.insert(0, '/Users/fanwei/study/HKV/hashemb')

import time
import numpy as np
import torch
import hashemb._hashemb_cpp as hcpp
from hashemb import HashEmbedding

# ============================================================================
# Config
# ============================================================================
BATCH_SIZE = 4096
FEATS_PER_SAMPLE = 200
EMBEDDING_DIM = 16
NUM_UNIQUE = 100_000

print("=" * 70)
print("Forward Pass Profiling")
print("=" * 70)
print(f"Batch size: {BATCH_SIZE}")
print(f"Features per sample: {FEATS_PER_SAMPLE}")
print(f"Total lookups: {BATCH_SIZE * FEATS_PER_SAMPLE:,}")
print()

# ============================================================================
# Initialize and warmup
# ============================================================================
emb = HashEmbedding(EMBEDDING_DIM, 1_000_000_000, optimizer='adam', lr=0.001)

# Warmup with some keys
with torch.no_grad():
    warmup_ids = torch.from_numpy(np.random.randint(1, 1_000_000_000, size=NUM_UNIQUE, dtype=np.int64))
    _ = emb(warmup_ids)

print(f"Warmup entries: {emb.num_entries:,}")
print()

# ============================================================================
# Profile full forward pass
# ============================================================================
N_ITER = 5

times_full_forward = []

for i in range(N_ITER):
    # Generate batch on CPU (no CUDA)
    batch_ids = torch.from_numpy(
        np.random.randint(1, 1_000_000_000, size=(BATCH_SIZE, FEATS_PER_SAMPLE), dtype=np.int64)
    )

    # Full forward
    t0 = time.perf_counter()
    emb_out = emb(batch_ids)
    t1 = time.perf_counter()
    times_full_forward.append(t1 - t0)

print("=" * 70)
print("Timing Breakdown (avg over {} iterations)".format(N_ITER))
print("=" * 70)

avg_full_forward = np.mean(times_full_forward) * 1000

print(f"  Full forward: {avg_full_forward:.2f}ms")
print()

# ============================================================================
# Profile C++ lookup_and_gather only (on CPU)
# ============================================================================
print("=" * 70)
print("Detailed C++ Profiling (single iteration)")
print("=" * 70)

batch_ids = torch.from_numpy(
    np.random.randint(1, 1_000_000_000, size=(BATCH_SIZE, FEATS_PER_SAMPLE), dtype=np.int64)
).cpu()
keys_np = batch_ids.numpy().flatten()

t0 = time.perf_counter()
emb_np, slot_np = emb._table.lookup_and_gather(keys_np)
t1 = time.perf_counter()

print(f"  lookup_and_gather (C++): {(t1-t0)*1000:.2f}ms")
print(f"  Entries after lookup: {emb.num_entries:,}")
print()

# ============================================================================
# Profile per-key lookup latency
# ============================================================================
N_SAMPLES = 1000
times_single_lookup = []

for _ in range(N_SAMPLES):
    key = np.int64(np.random.randint(1, 1_000_000_000))
    t0 = time.perf_counter()
    emb._table.lookup_and_gather(np.array([key]))
    t1 = time.perf_counter()
    times_single_lookup.append(t1 - t0)

print("=" * 70)
print("Single Key Lookup (avg over {} samples)".format(N_SAMPLES))
print("=" * 70)
print(f"  Per-key lookup: {np.mean(times_single_lookup)*1000:.4f}ms")
print("=" * 70)