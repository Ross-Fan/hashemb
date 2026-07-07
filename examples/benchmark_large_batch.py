#!/usr/bin/env python3
"""
Performance benchmark for large batch training scenario.

Target scenario:
- Batch size: 4096
- Features per sample: 200 hash IDs
- Total lookups per batch: 4096 × 200 = 819,200
- Embedding dim: 16
- Optimizer: Adam
"""
import time
import numpy as np
import torch
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

# Keep capacity high enough to hold all entries across all batches,
# but not so large that the 16 memory buckets OOM.
# For reference: capacity=1B → ~34GB hash table alone.
CAPACITY = 20_000_000
HASH_ID_RANGE = 100_000_000
WARMUP_UNIQUE = 500_000
BLOCK_SIZE = 1_000_000

print("=" * 70)
print("Large Batch Performance Benchmark")
print("=" * 70)
print(f"Config:")
print(f"  Batch size:          {BATCH_SIZE:,}")
print(f"  Features per sample: {FEATS_PER_SAMPLE:,}")
print(f"  Total lookups/batch: {BATCH_SIZE * FEATS_PER_SAMPLE:,}")
print(f"  Embedding dim:       {EMBEDDING_DIM}")
print(f"  Optimizer:           {OPTIMIZER}")
print(f"  Capacity:            {CAPACITY:,}")
print(f"  Hash ID range:       {HASH_ID_RANGE:,}")
print(f"  Warmup unique IDs:   {WARMUP_UNIQUE:,}")
print(f"  Device:              {'cuda' if torch.cuda.is_available() else 'cpu'}")
print()

# ============================================================================
# Generate synthetic data
# ============================================================================
print("Generating synthetic data...")
np.random.seed(42)
all_ids = np.random.randint(
    1, HASH_ID_RANGE,
    size=(NUM_ITERATIONS + NUM_WARMUP, BATCH_SIZE, FEATS_PER_SAMPLE),
    dtype=np.int64,
)
print(f"  Generated {len(all_ids)} batches")
print()

# ============================================================================
# Initialize model & warm up
# ============================================================================
print("Initializing HashEmbedding...")
emb = HashEmbedding(
    embedding_dim=EMBEDDING_DIM,
    capacity=CAPACITY,
    optimizer=OPTIMIZER,
    lr=LR,
    block_size=BLOCK_SIZE,
)

with torch.no_grad():
    warmup_ids = torch.from_numpy(
        np.random.randint(1, HASH_ID_RANGE, size=WARMUP_UNIQUE, dtype=np.int64)
    )
    _ = emb(warmup_ids)
print(f"  Warmup entries: {emb.num_entries:,}")

# Warmup iterations (full forward+backward+step)
for i in range(NUM_WARMUP):
    batch_ids = torch.from_numpy(all_ids[i]).long()
    out = emb(batch_ids)
    loss = out.sum()
    loss.backward()
    emb.step()
print(f"  After warmup entries: {emb.num_entries:,}")
print()

# ============================================================================
# Benchmark
# ============================================================================
print("=" * 70)
print(f"Timing breakdown (avg over {NUM_ITERATIONS} iterations)")
print("=" * 70)

times = {"forward": [], "backward": [], "step": [], "total": []}

for i in range(NUM_ITERATIONS):
    idx = NUM_WARMUP + i
    batch_ids = torch.from_numpy(all_ids[idx]).long()

    # Forward
    t0 = time.perf_counter()
    out = emb(batch_ids)
    t1 = time.perf_counter()

    # Backward
    loss = out.sum()
    t2 = time.perf_counter()
    loss.backward()
    t3 = time.perf_counter()

    # Step
    emb.step()
    t4 = time.perf_counter()

    times["forward"].append(t1 - t0)
    times["backward"].append(t3 - t2)
    times["step"].append(t4 - t3)
    times["total"].append(t4 - t0)

    if i < 3:
        print(f"  Iter {i+1}: fwd={times['forward'][-1]*1000:.1f}ms  "
              f"bwd={times['backward'][-1]*1000:.1f}ms  "
              f"step={times['step'][-1]*1000:.1f}ms  "
              f"total={times['total'][-1]*1000:.1f}ms")

print()
print("-" * 70)
for name in ["forward", "backward", "step", "total"]:
    arr = np.array(times[name]) * 1000
    print(f"  {name:10s}: mean={arr.mean():8.2f}ms  std={arr.std():6.2f}ms  "
          f"min={arr.min():8.2f}ms  max={arr.max():8.2f}ms")

total_lookups = BATCH_SIZE * FEATS_PER_SAMPLE * NUM_ITERATIONS
total_sec = sum(times["total"])
throughput = total_lookups / total_sec
print()
print(f"  Throughput: {throughput:,.0f} keys/sec  ({throughput/1e6:.1f}M keys/sec)")
print(f"  Final entries: {emb.num_entries:,}")
print("=" * 70)
