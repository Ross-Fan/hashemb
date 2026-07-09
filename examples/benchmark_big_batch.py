#!/usr/bin/env python3
"""
Big-batch HashEmb stability / stress test.

Tests HashEmb under sustained training with a LARGE table (entries grow
over time as new feat IDs are encountered).  Monitors memory growth and
per-batch timing to detect performance degradation.

Scenario:
  - Batch size: 4096, 100 feat IDs per sample
  - Feat IDs from [1, 100M] hash space (too large for nn.Embedding)
  - 500K unique feat IDs in the pool → hash table grows to ~500K entries
  - Binary labels (CTR-like), learnable signal from 2K signal features
  - Long training: 100 epochs by default

Memory monitoring per epoch:
  - Current RSS (MB) and delta from previous epoch
  - Table size (num_entries) and growth rate
  - Estimated table memory vs measured RSS — warns on unexplained bloat

Stability checks at end:
  - NaN/Inf in weights
  - Throughput trend (degradation as table grows?)
  - Memory per entry (stable or leaking?)
  - Continuous block allocation (block_count)

Usage:
    # Default: 500K unique feat IDs, 100 epochs, batch=4096
    python examples/benchmark_big_batch.py

    # Heavy stress: 2M feat IDs, 500 epochs, small block → many expansions
    python examples/benchmark_big_batch.py --unique-feats 2000000 --steps 500 --block-size 100000

    # Time-based: run for ~10 minutes
    python examples/benchmark_big_batch.py --duration 600

    # Ultra-large hash space test
    python examples/benchmark_big_batch.py --hash-range 1000000000
"""

import argparse
import resource
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

from hashemb import HashEmbedding

# =============================================================================
# Config
# =============================================================================
BATCH_SIZE = 4096
FEATS_PER_SAMPLE = 100
EMBEDDING_DIM = 16
N_UNIQUE_FEATS = 500_000            # unique feat IDs — key driver of table growth
N_SIGNAL_FEATS = 2_000              # these determine the label
HASH_ID_RANGE = 100_000_000         # 100M hash space
N_TRAIN = 80_000                    # training samples per "epoch" (cycling)
N_VAL = 10_000                      # validation samples
EPOCHS = 100
LR = 0.01
HASH_CAPACITY = 2_000_000           # large enough for all unique IDs
BLOCK_SIZE = 200_000                # smallish blocks → exercise expansion path
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# =============================================================================
# Helpers
# =============================================================================
def mem_rss_mb():
    """Return current RSS in MB (Linux: ru_maxrss in KB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


# =============================================================================
# Synthetic data
# =============================================================================
def generate_synthetic_data(n_unique, n_signal, hash_range, n_train, n_val,
                            feats_per_sample, emb_dim, seed):
    """Generate label-bearing data.  Signal features (first n_signal) determine
    labels via fixed ground-truth embeddings + linear model.  Other features are
    noise."""
    rng = np.random.RandomState(seed)

    # Unique raw feat IDs sampled sparsely from hash_range
    raw_ids = rng.choice(hash_range, size=n_unique, replace=False).astype(np.int64)

    # Ground-truth: signal features have meaningful embeddings
    gt_emb = rng.randn(n_unique, emb_dim).astype(np.float32) * 0.1
    gt_w = rng.randn(1, emb_dim).astype(np.float32) * 0.1
    gt_b = float(rng.randn(1).astype(np.float32)[0] * 0.01)

    def _gen(n):
        idx = rng.randint(0, n_unique, size=(n, feats_per_sample), dtype=np.int64)
        labels = []
        for i in range(n):
            row = idx[i]
            sig = row[row < n_signal]
            if len(sig) == 0:
                p = 0.5
            else:
                pooled = gt_emb[sig].mean(axis=0)
                score = float(np.dot(pooled, gt_w[0]) + gt_b)
                p = 1.0 / (1.0 + np.exp(-score))
            labels.append(1 if rng.random() < p else 0)
        return raw_ids[idx].astype(np.int64), np.array(labels, dtype=np.float32)

    return _gen(n_train), _gen(n_val), n_unique


# =============================================================================
# Dataset
# =============================================================================
class SparseFeatDataset(Dataset):
    def __init__(self, feat_ids, labels):
        self.feat_ids = feat_ids
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.feat_ids[idx]).long(),
                torch.tensor(self.labels[idx], dtype=torch.float32))


def collate_fn(batch):
    ids = torch.stack([b[0] for b in batch])
    lbl = torch.tensor([b[1].item() for b in batch], dtype=torch.float32)
    return ids, lbl


# =============================================================================
# Model
# =============================================================================
class BigBatchModel(torch.nn.Module):
    def __init__(self, emb_dim, capacity, lr, block_size):
        super().__init__()
        self.emb = HashEmbedding(emb_dim, capacity,
                                 optimizer="adam", lr=lr,
                                 initial_scale=0.01,
                                 block_size=block_size)
        self.predict = torch.nn.Linear(emb_dim, 1)
        torch.nn.init.xavier_uniform_(self.predict.weight)
        torch.nn.init.zeros_(self.predict.bias)

    def forward(self, feat_ids):
        embs = self.emb(feat_ids)
        pooled = embs.mean(dim=1)
        return self.predict(pooled).squeeze(-1)

    def step(self):
        self.emb.step()


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="HashEmb stability / stress test")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--steps", type=int, default=EPOCHS)
    parser.add_argument("--duration", type=int, default=0,
                        help="Run for N seconds (overrides --steps)")
    parser.add_argument("--unique-feats", type=int, default=N_UNIQUE_FEATS)
    parser.add_argument("--hash-range", type=int, default=HASH_ID_RANGE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--capacity", type=int, default=HASH_CAPACITY)
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    batch_size = args.batch_size
    epochs = args.steps if args.duration <= 0 else 999999
    duration = args.duration
    n_unique = args.unique_feats
    hash_range = args.hash_range
    lr = args.lr
    capacity = args.capacity
    block_size = args.block_size

    n_batches = N_TRAIN // batch_size
    lookups_per_batch = batch_size * FEATS_PER_SAMPLE
    total_lookups_per_epoch = lookups_per_batch * n_batches

    # ── Estimate scale ──
    bytes_per_entry = EMBEDDING_DIM * 4 * 4  # Adam: weight,grad,m,v = 4 × float32
    est_table_mb = n_unique * bytes_per_entry / (1024 ** 2)
    est_blocks = max(1, (n_unique + block_size - 1) // block_size)
    print("=" * 70)
    print("HashEmb Stability / Stress Test")
    print("=" * 70)
    print(f"  Batch size:            {batch_size:,}")
    print(f"  Feats per sample:      {FEATS_PER_SAMPLE}")
    print(f"  Lookups / batch:       {lookups_per_batch:,}")
    print(f"  Lookups / epoch:       {total_lookups_per_epoch:,}")
    print(f"  Embedding dim:         {EMBEDDING_DIM}")
    print(f"  Hash ID range:         [1, {hash_range:,}]")
    print(f"  Unique feat IDs:       {n_unique:,}")
    print(f"  Signal features:       {N_SIGNAL_FEATS}")
    print(f"  Est table memory:      {est_table_mb:.0f} MB ({n_unique:,} × {bytes_per_entry}B)")
    print(f"  Block size:            {block_size:,} → ~{est_blocks} blocks expected")
    print(f"  Capacity hint:         {capacity:,}")
    print(f"  Training samples:      {N_TRAIN:,}")
    print(f"  Batches / epoch:       {n_batches}")
    print(f"  Epochs:                {epochs if not duration else 'until timeout'}")
    if duration:
        print(f"  Duration limit:        {duration}s")
    print(f"  LR:                    {lr}")
    print(f"  Debug:                 {args.debug}")
    print()

    # =========================================================================
    # Data
    # =========================================================================
    mem0 = mem_rss_mb()
    print(f"[MEM] Baseline:  {mem0:.0f} MB")
    t0 = time.time()

    (train_ids, train_labels), (val_ids, val_labels), actual_unique = \
        generate_synthetic_data(
            n_unique, N_SIGNAL_FEATS, hash_range, N_TRAIN, N_VAL,
            FEATS_PER_SAMPLE, EMBEDDING_DIM, SEED)

    t_data = time.time() - t0
    mem1 = mem_rss_mb()
    print(f"  Data:         {n_unique:,} unique IDs ({N_TRAIN:,} train + {N_VAL:,} val) "
          f"→ {t_data:.1f}s")
    print(f"  Label balance: train={train_labels.sum():.0f}/{len(train_labels)} "
          f"({100*train_labels.sum()/len(train_labels):.1f}%), "
          f"val={val_labels.sum():.0f}/{len(val_labels)} "
          f"({100*val_labels.sum()/len(val_labels):.1f}%)")
    print(f"[MEM] +data:     {mem1:.0f} MB  (+{mem1 - mem0:.0f} MB)")
    print()

    train_ds = SparseFeatDataset(train_ids, train_labels)
    val_ds = SparseFeatDataset(val_ids, val_labels)
    train_loader = DataLoader(train_ds, batch_size, shuffle=True,
                              collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size, shuffle=False,
                            collate_fn=collate_fn, drop_last=False)

    # =========================================================================
    # Model
    # =========================================================================
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    mem2 = mem_rss_mb()
    print(f"[MEM] Before model: {mem2:.0f} MB")

    model = BigBatchModel(EMBEDDING_DIM, capacity, lr, block_size)
    opt = torch.optim.Adam(model.predict.parameters(), lr=lr)

    mem3 = mem_rss_mb()
    print(f"  Initial entries: {model.emb.num_entries:,}")
    print(f"[MEM] +model:        {mem3:.0f} MB  (+{mem3 - mem2:.0f} MB)")
    print()

    # =========================================================================
    # Training with per-epoch monitoring
    # =========================================================================
    # Snapshots for stability analysis
    snap_mem = []       # (epoch, rss_mb, num_entries)
    snap_timing = []    # (epoch, fwd_ms, bwd_ms, step_ms, total_ms)
    snap_auc = []       # (epoch, auc)
    prev_mem = mem3
    prev_ent = 0
    wall_start = time.time()

    print("-" * 90)
    print(f"{'Ep':>4s} | {'loss':>7s} {'auc':>7s} | "
          f"{'entries':>10s} {'RSS(MB)':>8s} {'ΔMB':>6s} | "
          f"{'fwd':>6s} {'bwd':>6s} {'step':>6s} {'total':>6s} | {'Mlookup/s':>9s}")
    print("-" * 90)

    epoch = 0
    while True:
        epoch += 1
        if epoch > epochs:
            break
        if duration and (time.time() - wall_start) > duration:
            break

        model.train()
        ep_loss = 0.0
        ep_fwd, ep_bwd, ep_step, ep_tot = [], [], [], []

        for bi, (batch_ids, labels) in enumerate(train_loader):
            t0 = time.perf_counter()
            opt.zero_grad()
            logits = model(batch_ids)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            t1 = time.perf_counter()

            loss.backward()
            t2 = time.perf_counter()

            opt.step()
            model.step()
            t3 = time.perf_counter()

            ep_loss += loss.item()
            ep_fwd.append(t1 - t0)
            ep_bwd.append(t2 - t1)
            ep_step.append(t3 - t2)
            ep_tot.append(t3 - t0)

        # ── Validation ──
        model.eval()
        all_scores, all_labels_gt = [], []
        with torch.no_grad():
            for batch_ids, labels in val_loader:
                logits = model(batch_ids)
                all_scores.append(torch.sigmoid(logits).numpy())
                all_labels_gt.append(labels.numpy())
        auc = roc_auc_score(np.concatenate(all_labels_gt),
                            np.concatenate(all_scores))

        # ── Metrics ──
        n_ent = model.emb.num_entries
        cur_mem = mem_rss_mb()
        mem_delta = cur_mem - prev_mem
        ent_delta = n_ent - prev_ent

        avg_fwd = np.mean(ep_fwd) * 1000
        avg_bwd = np.mean(ep_bwd) * 1000
        avg_step = np.mean(ep_step) * 1000
        avg_tot = np.mean(ep_tot) * 1000
        tput = lookups_per_batch / (avg_tot / 1000) / 1e6

        snap_mem.append((epoch, cur_mem, n_ent))
        snap_timing.append((epoch, avg_fwd, avg_bwd, avg_step, avg_tot))
        snap_auc.append((epoch, auc))

        print(f"{epoch:4d} | {ep_loss/n_batches:7.4f} {auc:7.4f} | "
              f"{n_ent:10,d} {cur_mem:8.0f} {mem_delta:+6.0f} | "
              f"{avg_fwd:6.1f} {avg_bwd:6.1f} {avg_step:6.1f} {avg_tot:6.1f} | "
              f"{tput:9.1f}")

        # ── Detailed snapshot every 10 epochs ──
        if args.debug and epoch % 10 == 0:
            print(f"  [DEBUG ep{epoch}] ent_growth=+{ent_delta:,}  "
                  f"mb_per_1k_ent={mem_delta/(ent_delta+1)*1000:.2f}  "
                  f"blocks≈{n_ent//block_size + 1}  "
                  f"table_fill={100*n_ent/n_unique:.1f}%")

        prev_mem = cur_mem
        prev_ent = n_ent

    total_time = time.time() - wall_start

    # =========================================================================
    # Summary
    # =========================================================================
    print("-" * 90)
    print()
    print("=" * 70)
    print("Stability Summary")
    print("=" * 70)
    print(f"  Total epochs:   {epoch}")
    print(f"  Total time:     {total_time:.1f}s ({total_time/epoch:.1f}s/epoch)")
    print(f"  Total batches:  {epoch * n_batches:,}")
    print(f"  Total lookups:  {epoch * total_lookups_per_epoch:,}")
    print()

    # ── Entry / memory growth ──
    if len(snap_mem) >= 2:
        ent_first = snap_mem[0][2]
        ent_last = snap_mem[-1][2]
        mem_first = snap_mem[0][1]
        mem_last = snap_mem[-1][1]
        ent_growth = ent_last - ent_first
        mem_growth = mem_last - mem_first
        mb_per_kent = (mem_growth / (ent_growth + 1)) * 1000
        theoretical_b_per_entry = bytes_per_entry * 1000  # bytes per 1k entries

        # Table fill %
        fill_pct = 100.0 * ent_last / n_unique

        print("Entry & Memory Growth:")
        print(f"  Entries:  {ent_first:>10,} → {ent_last:>10,}  "
              f"(+{ent_growth:,}, {fill_pct:.1f}% of pool)")
        print(f"  RSS:      {mem_first:>10.0f} → {mem_last:>10.0f} MB  "
              f"(+{mem_growth:.0f} MB)")
        print(f"  MB / k entries: {mb_per_kent:.2f}  "
              f"(theoretical: {bytes_per_entry * 1000 / (1024**2):.2f} MB/k for Adam 4× buffers)")
        print()

    # ── Timing stability ──
    if len(snap_timing) >= 4:
        print("Timing Stability (first 3 vs last 3 epochs):")
        for idx, label in [(1, "fwd"), (2, "bwd"), (3, "step"), (4, "total")]:
            first3 = np.mean([s[idx] for s in snap_timing[:3]])
            last3 = np.mean([s[idx] for s in snap_timing[-3:]])
            delta = last3 - first3
            delta_pct = 100 * delta / (first3 + 0.001)
            flag = ""
            if delta_pct > 5:
                flag = " ⚠ +{:.0f}% slow".format(delta_pct)
            elif delta_pct < -5:
                flag = " (faster)"
            print(f"  {label:6s}: {first3:6.1f}ms → {last3:6.1f}ms  "
                  f"(Δ={delta:+.1f}ms, {delta_pct:+.1f}%){flag}")
        print()

    # ── AUC ──
    if snap_auc:
        aucs = [a for _, a in snap_auc]
        print(f"AUC:  start={aucs[0]:.4f}  best={max(aucs):.4f}  "
              f"final={aucs[-1]:.4f}  "
              f"{'▲' if aucs[-1] > aucs[0] else '▼' if aucs[-1] < aucs[0] else '—'}")
        print()

    # ── Weight sanity ──
    sd = model.emb.state_dict()
    w = sd["weight"]
    nan_count = torch.isnan(w).sum().item()
    inf_count = torch.isinf(w).sum().item()
    w_min = w.min().item()
    w_max = w.max().item()
    w_mean = w.mean().item()
    w_std = w.std().item()

    if nan_count > 0 or inf_count > 0:
        print(f"✗ WEIGHT ISSUE: {nan_count} NaN, {inf_count} Inf detected!")
    else:
        print(f"✓ Weights OK:  min={w_min:+.4f}  max={w_max:+.4f}  "
              f"mean={w_mean:+.6f}  std={w_std:+.4f}")

    # ── Overall stability verdict ──
    print()
    issues = []
    if nan_count > 0 or inf_count > 0:
        issues.append("NaN/Inf in weights")
    if ent_growth <= 0:
        issues.append("No entry growth — all IDs seen in first epoch? Try larger --unique-feats")
    if len(snap_timing) >= 4:
        first3_total = np.mean([s[4] for s in snap_timing[:3]])
        last3_total = np.mean([s[4] for s in snap_timing[-3:]])
        slowdown_pct = 100 * (last3_total - first3_total) / (first3_total + 0.001)
        if slowdown_pct > 20:
            issues.append(f"timing degraded {slowdown_pct:.0f}%")
        elif slowdown_pct > 10:
            issues.append(f"timing degraded {slowdown_pct:.0f}% (moderate)")
    if mb_per_kent > theoretical_b_per_entry * 2:
        issues.append(f"memory/entry 2× above theoretical")

    if not issues:
        print("✓ STABILITY PASS — no issues detected")
    else:
        for issue in issues:
            print(f"⚠ {issue}")
    print("=" * 70)


if __name__ == "__main__":
    main()
