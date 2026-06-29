#!/usr/bin/env python3
"""
Performance benchmark for large batch training scenario.

Target scenario (from user):
- Batch size: 4096
- Features per sample: 200 hash IDs
- Total lookups per batch: 4096 × 200 = 819,200
- Embedding dim: 16
- Optimizer: Adam
- Hash ID space: ~1B (simulated with random IDs)
"""
import sys
sys.path.insert(0, '/Users/fanwei/study/HKV/hashemb')

import time
import numpy as np
import torch
import torch.nn as nn
from hashemb import HashEmbedding

# ============================================================================
# Config
# ============================================================================
BATCH_SIZE = 4096
FEATS_PER_SAMPLE = 200
EMBEDDING_DIM = 16
OPTIMIZER = "adam"
LR = 0.001
NUM_WARMUP = 3
NUM_ITERATIONS = 10
HASH_ID_RANGE = 1_000_000_000  # 1B
EXPECTED_UNIQUE_IDS = 100_000  # Reduce for faster warmup

print("=" * 70)
print("Large Batch Performance Benchmark")
print("=" * 70)
print(f"Config:")
print(f"  Batch size:          {BATCH_SIZE:,}")
print(f"  Features per sample: {FEATS_PER_SAMPLE:,}")
print(f"  Total lookups/batch: {BATCH_SIZE * FEATS_PER_SAMPLE:,}")
print(f"  Embedding dim:       {EMBEDDING_DIM}")
print(f"  Optimizer:           {OPTIMIZER}")
print(f"  Hash ID range:       {HASH_ID_RANGE:,}")
print(f"  Expected unique IDs: {EXPECTED_UNIQUE_IDS:,}")
print()

# ============================================================================
# Generate synthetic data
# ============================================================================
print("Generating synthetic data...")

# Generate random hash IDs (simulating already-hashed feature IDs)
# Using numpy for fast generation
all_ids = np.random.randint(1, HASH_ID_RANGE, size=(NUM_ITERATIONS + NUM_WARMUP, BATCH_SIZE, FEATS_PER_SAMPLE), dtype=np.int64)

print(f"  Generated {len(all_ids)} batches")
print(f"  Unique IDs in first 5 batches: {len(np.unique(all_ids[:5].flatten())):,}")
print()

# ============================================================================
# Initialize model
# ============================================================================
print("Initializing HashEmbedding...")

emb = HashEmbedding(
    embedding_dim=EMBEDDING_DIM,
    capacity=HASH_ID_RANGE,
    optimizer=OPTIMIZER,
    lr=LR,
    block_size=1_000_000,  # Use new default
)

print(f"  Initial entries: {emb.num_entries:,}")
print()

# ============================================================================
# Warm up - populate with some IDs to simulate real training
# ============================================================================
print("Warming up (populating hash table)...")

with torch.no_grad():
    warmup_ids = torch.from_numpy(np.random.randint(1, HASH_ID_RANGE, size=(EXPECTED_UNIQUE_IDS,), dtype=np.int64))
    _ = emb(warmup_ids)  # Create entries

print(f"  Warmup entries: {emb.num_entries:,}")
print()

# ============================================================================
# Benchmark
# ============================================================================
print("=" * 70)
print("Benchmark")
print("=" * 70)

forward_times = []
backward_times = []
step_times = []
total_times = []

for i in range(NUM_ITERATIONS + NUM_WARMUP):
    # Prepare batch
    batch_ids = torch.from_numpy(all_ids[i]).long()  # (BATCH_SIZE, FEATS_PER_SAMPLE)

    # Forward
    t0 = time.perf_counter()
    embeddings = emb(batch_ids)  # (BATCH_SIZE, FEATS_PER_SAMPLE, EMBEDDING_DIM)
    t1 = time.perf_counter()
    forward_time = t1 - t0

    # Simple dense layer for gradient computation
    # Flatten: (BATCH_SIZE, FEATS_PER_SAMPLE, EMBEDDING_DIM) -> (BATCH_SIZE, FEATS_PER_SAMPLE * EMBEDDING_DIM)
    flat_emb = embeddings.view(BATCH_SIZE, -1)
    dense_out = torch.randn(BATCH_SIZE, 1, requires_grad=True)  # Dummy dense output
    loss = (flat_emb.sum() + dense_out.sum())  # Dummy loss

    # Backward
    t2 = time.perf_counter()
    loss.backward()
    t3 = time.perf_counter()
    backward_time = t3 - t2

    # Step (optimizer update)
    t4 = time.perf_counter()
    emb.step()
    t5 = time.perf_counter()
    step_time = t5 - t4

    total_time = (t1 - t0) + (t3 - t2) + (t5 - t4)

    # Skip warmup iterations
    if i >= NUM_WARMUP:
        forward_times.append(forward_time)
        backward_times.append(backward_time)
        step_times.append(step_time)
        total_times.append(total_time)

        if (i - NUM_WARMUP) < 3:  # Print first 3 iterations
            print(f"  Iter {i-NUM_WARMUP+1:2d}: "
                  f"forward={forward_time*1000:6.2f}ms, "
                  f"backward={backward_time*1000:6.2f}ms, "
                  f"step={step_time*1000:6.2f}ms, "
                  f"total={total_time*1000:6.2f}ms")

# ============================================================================
# Results
# ============================================================================
print()
print("=" * 70)
print("Results (averaged over {} iterations)".format(NUM_ITERATIONS))
print("=" * 70)

def print_stats(name, times):
    times_ms = np.array(times) * 1000
    print(f"  {name:12s}: mean={times_ms.mean():7.2f}ms, "
          f"std={times_ms.std():5.2f}ms, "
          f"min={times_ms.min():6.2f}ms, "
          f"max={times_ms.max():6.2f}ms")

print_stats("Forward", forward_times)
print_stats("Backward", backward_times)
print_stats("Step", step_times)
print_stats("Total", total_times)

print()
print("Breakdown:")
total_mean = np.mean(total_times)
print(f"  Forward:  {np.mean(forward_times) / total_mean * 100:5.1f}%")
print(f"  Backward: {np.mean(backward_times) / total_mean * 100:5.1f}%")
print(f"  Step:     {np.mean(step_times) / total_mean * 100:5.1f}%")

print()
print("Throughput:")
total_lookups = BATCH_SIZE * FEATS_PER_SAMPLE * NUM_ITERATIONS
total_time_sec = sum(total_times)
throughput = total_lookups / total_time_sec
print(f"  {throughput:,.0f} lookups/sec")
print(f"  {throughput / 1e6:,.1f}M lookups/sec")

print()
print(f"Final table entries: {emb.num_entries:,}")
print()
print("=" * 70)