#!/usr/bin/env python3
"""
Big-batch HashEmb benchmark at scale.

Scenario:
  - Batch size: 4096, 100 feat IDs per sample
  - Feat IDs drawn from [1, 100M] hash space (too large for nn.Embedding)
  - Binary labels (CTR-like), generated from learnable ground-truth embeddings
  - Tests: correctness, per-batch timing, memory scaling under sustained training

Synthetic data:
  - 10K unique raw feat IDs randomly sampled from [1, 100M]
  - Each sample picks 100 feat IDs from the pool
  - 200 "signal" features determine labels via GT embeddings + linear model
  - 80K training + 20K validation samples

Model: (B, 100) feat IDs → HashEmbedding → mean pool → Linear(1) → logit

Usage:
    python examples/benchmark_big_batch.py
    python examples/benchmark_big_batch.py --debug --steps 50
    python examples/benchmark_big_batch.py --batch-size 8192
    python examples/benchmark_big_batch.py --capacity 500000
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
N_UNIQUE_FEATS = 10_000          # unique feat IDs in the data pool
N_SIGNAL_FEATS = 200             # these determine the label
HASH_ID_RANGE = 100_000_000       # 100M hash space
N_TRAIN = 80_000                  # training samples
N_VAL = 20_000                    # validation samples
EPOCHS = 20
LR = 0.01
HASH_CAPACITY = 100_000
BLOCK_SIZE = 1_000_000
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# =============================================================================
# Helpers
# =============================================================================
def mem_rss_mb():
    """Return current RSS in MB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def fmt_rate(val, total_sec):
    """Human-readable rate: val/total_sec."""
    if total_sec <= 0:
        return "?"
    rate = val / total_sec
    if rate >= 1e6:
        return f"{rate / 1e6:.1f}M/s"
    elif rate >= 1e3:
        return f"{rate / 1e3:.1f}K/s"
    return f"{rate:.1f}/s"


# =============================================================================
# Synthetic data generation
# =============================================================================
def generate_synthetic_data():
    """
    Generate CTR-like data with learnable signal.

    Returns:
        train_ids:  (N_TRAIN, FEATS_PER_SAMPLE) — raw feat IDs (sparse 1..100M)
        train_labels: (N_TRAIN,) — binary
        val_ids:    (N_VAL, FEATS_PER_SAMPLE)
        val_labels: (N_VAL,)
    """
    rng = np.random.RandomState(SEED)

    # 10K unique raw feat IDs from [1, 100M)
    raw_feat_ids = rng.choice(HASH_ID_RANGE, size=N_UNIQUE_FEATS, replace=False).astype(np.int64)

    # Ground-truth embeddings for signal features (first N_SIGNAL_FEATS)
    gt_emb = rng.randn(N_UNIQUE_FEATS, EMBEDDING_DIM).astype(np.float32) * 0.1
    gt_w = rng.randn(1, EMBEDDING_DIM).astype(np.float32) * 0.1
    gt_b = float(rng.randn(1).astype(np.float32)[0] * 0.01)

    def _gen(n_samples):
        indices = rng.randint(0, N_UNIQUE_FEATS, size=(n_samples, FEATS_PER_SAMPLE), dtype=np.int64)
        labels = []
        for i in range(n_samples):
            row = indices[i]
            signal_in_row = row[row < N_SIGNAL_FEATS]
            if len(signal_in_row) == 0:
                p = 0.5
            else:
                pooled = gt_emb[signal_in_row].mean(axis=0)
                score = float(np.dot(pooled, gt_w[0]) + gt_b)
                p = 1.0 / (1.0 + np.exp(-score))
            labels.append(1 if rng.random() < p else 0)

        raw_ids = raw_feat_ids[indices].astype(np.int64)
        return raw_ids, np.array(labels, dtype=np.float32)

    train_ids, train_labels = _gen(N_TRAIN)
    val_ids, val_labels = _gen(N_VAL)
    return train_ids, train_labels, val_ids, val_labels


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
    def __init__(self, emb_dim, capacity, lr):
        super().__init__()
        self.emb = HashEmbedding(emb_dim, capacity,
                                 optimizer="adam", lr=lr,
                                 initial_scale=0.01,
                                 block_size=BLOCK_SIZE)
        self.predict = torch.nn.Linear(emb_dim, 1)
        torch.nn.init.xavier_uniform_(self.predict.weight)
        torch.nn.init.zeros_(self.predict.bias)

    def forward(self, feat_ids):
        # (B, F) → (B, F, D) → mean pool → (B, D) → (B,)
        embs = self.emb(feat_ids)
        pooled = embs.mean(dim=1)
        return self.predict(pooled).squeeze(-1)

    def step(self):
        self.emb.step()


# =============================================================================
# Main
# =============================================================================
def main():
    # ── Args ──
    parser = argparse.ArgumentParser(description="Big-batch HashEmb benchmark")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--steps", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--capacity", type=int, default=HASH_CAPACITY)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--timing-warmup", type=int, default=3,
                        help="Batches to warm-up before timing (per epoch)")
    args = parser.parse_args()

    batch_size = args.batch_size
    epochs = args.steps
    lr = args.lr
    capacity = args.capacity
    n_batches = N_TRAIN // batch_size

    # =========================================================================
    # Header
    # =========================================================================
    print("=" * 70)
    print("Big-Batch HashEmb Benchmark")
    print("=" * 70)
    print(f"  Batch size:           {batch_size:,}")
    print(f"  Feats per sample:     {FEATS_PER_SAMPLE}")
    print(f"  Total lookups/batch:  {batch_size * FEATS_PER_SAMPLE:,}")
    print(f"  Embedding dim:        {EMBEDDING_DIM}")
    print(f"  Hash ID range:        [1, {HASH_ID_RANGE:,}]")
    print(f"  Unique feat IDs:      {N_UNIQUE_FEATS:,}")
    print(f"  Signal features:      {N_SIGNAL_FEATS}")
    print(f"  Training samples:     {N_TRAIN:,}")
    print(f"  Validation samples:   {N_VAL:,}")
    print(f"  Batches/epoch:        {n_batches}")
    print(f"  Epochs:               {epochs}")
    print(f"  LR:                   {lr}")
    print(f"  Capacity:             {capacity:,}")
    print(f"  Block size:           {BLOCK_SIZE:,}")
    print(f"  Debug:                {args.debug}")
    print()

    # =========================================================================
    # Data
    # =========================================================================
    mem0 = mem_rss_mb()
    print(f"[MEM] Before data gen: {mem0:.1f} MB")
    t0 = time.time()

    train_ids, train_labels, val_ids, val_labels = generate_synthetic_data()

    t_data = time.time() - t0
    mem1 = mem_rss_mb()
    print(f"  Data generated in {t_data:.1f}s")
    print(f"  Label balance: train={train_labels.sum():.0f}/{len(train_labels)} "
          f"({100*train_labels.sum()/len(train_labels):.1f}%), "
          f"val={val_labels.sum():.0f}/{len(val_labels)} "
          f"({100*val_labels.sum()/len(val_labels):.1f}%)")
    print(f"[MEM] After data gen:  {mem1:.1f} MB  (+{mem1 - mem0:.1f} MB)")
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
    print(f"[MEM] Before model:    {mem2:.1f} MB")
    print("Creating HashEmbedding...")

    model = BigBatchModel(EMBEDDING_DIM, capacity, lr)
    opt = torch.optim.Adam(model.predict.parameters(), lr=lr)

    mem3 = mem_rss_mb()
    print(f"  Initial entries:     {model.emb.num_entries:,}")
    print(f"[MEM] After model:     {mem3:.1f} MB  (+{mem3 - mem2:.1f} MB)")
    print()

    # =========================================================================
    # Training with per-batch timing
    # =========================================================================
    # We collect timing for all batches (after warmup) to compute stats
    all_times = {"fwd": [], "bwd": [], "step": [], "total": []}
    all_aucs = []
    all_ent = []

    total_t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0

        # Per-batch timing for this epoch
        ep_fwd = []
        ep_bwd = []
        ep_step = []
        ep_tot = []

        for bi, (batch_ids, labels) in enumerate(train_loader):
            # Forward
            t0 = time.perf_counter()
            opt.zero_grad()
            logits = model(batch_ids)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            t1 = time.perf_counter()

            # Backward
            loss.backward()
            t2 = time.perf_counter()

            # Step: dense + hash
            opt.step()
            model.step()
            t3 = time.perf_counter()

            ep_loss += loss.item()

            # Collect timing (skip first N batches for warmup)
            if bi >= args.timing_warmup:
                ep_fwd.append(t1 - t0)
                ep_bwd.append(t2 - t1)
                ep_step.append(t3 - t2)
                ep_tot.append(t3 - t0)

            if args.debug and bi == 0:
                print(f"  [ep{epoch:2d} batch 0] "
                      f"loss={loss.item():.4f}  "
                      f"fwd={1000*(t1-t0):.1f}ms  "
                      f"bwd={1000*(t2-t1):.1f}ms  "
                      f"step={1000*(t3-t2):.1f}ms  "
                      f"mem={mem_rss_mb():.0f}MB")

        # Validation
        model.eval()
        all_scores, all_labels_gt = [], []
        with torch.no_grad():
            for batch_ids, labels in val_loader:
                logits = model(batch_ids)
                all_scores.append(torch.sigmoid(logits).numpy())
                all_labels_gt.append(labels.numpy())

        auc = roc_auc_score(np.concatenate(all_labels_gt),
                            np.concatenate(all_scores))
        n_ent = model.emb.num_entries
        all_aucs.append(auc)
        all_ent.append(n_ent)

        # Aggregate timing for this epoch
        if ep_tot:
            avg_fwd = np.mean(ep_fwd) * 1000
            avg_bwd = np.mean(ep_bwd) * 1000
            avg_step = np.mean(ep_step) * 1000
            avg_tot = np.mean(ep_tot) * 1000
            # Lookup throughput
            lookups_per_batch = batch_size * FEATS_PER_SAMPLE
            throughput = lookups_per_batch / np.mean(ep_tot)

            all_times["fwd"].append(avg_fwd)
            all_times["bwd"].append(avg_bwd)
            all_times["step"].append(avg_step)
            all_times["total"].append(avg_tot)

            print(f"  Ep {epoch:2d}: loss={ep_loss/n_batches:.4f}  "
                  f"auc={auc:.4f}  entries={n_ent:,}  "
                  f"[fwd={avg_fwd:.1f}ms bwd={avg_bwd:.1f}ms "
                  f"step={avg_step:.1f}ms total={avg_tot:.1f}ms]  "
                  f"({throughput/1e6:.1f}M lookup/s)  "
                  f"mem={mem_rss_mb():.0f}MB")
        else:
            print(f"  Ep {epoch:2d}: loss={ep_loss/n_batches:.4f}  "
                  f"auc={auc:.4f}  entries={n_ent:,}  "
                  f"(all batches used for warmup)")

    total_time = time.time() - total_t0

    # =========================================================================
    # Summary
    # =========================================================================
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)

    best_auc = max(all_aucs)
    print(f"  Best AUC:            {best_auc:.4f} (epoch {all_aucs.index(best_auc)+1})")
    print(f"  Final AUC:           {all_aucs[-1]:.4f}")
    print(f"  AUC trend:           {'▲' if all_aucs[-1] > all_aucs[0] else '▼'} "
          f"{all_aucs[0]:.4f} → {all_aucs[-1]:.4f}")
    print()

    # Per-batch timing stats (aggregated across all timed epochs)
    if all_times["total"]:
        print("Per-batch timing (mean ± std across epochs, after warmup):")
        for k in ["fwd", "bwd", "step", "total"]:
            arr = all_times[k]
            print(f"  {k:6s}: {np.mean(arr):6.1f} ± {np.std(arr):5.1f} ms  "
                  f"(min={np.min(arr):.1f}, max={np.max(arr):.1f})")

        lookups_per_batch = batch_size * FEATS_PER_SAMPLE
        avg_total_ms = np.mean(all_times["total"])
        throughput = lookups_per_batch / (avg_total_ms / 1000)
        print(f"  Throughput: {throughput/1e6:.1f}M lookups/s "
              f"({lookups_per_batch * n_batches / total_time / 1e6:.1f}M/s overall)")
    print()

    # Memory
    mem_final = mem_rss_mb()
    print(f"Memory:")
    print(f"  Before:  {mem0:.0f} MB")
    print(f"  Peak:    {max(mem_final, max([mem0, mem1, mem2, mem3])):.0f} MB")
    print(f"  After:   {mem_final:.0f} MB  (+{mem_final - mem2:.0f} MB from model init)")
    print(f"  Entries: {all_ent[0]:,} → {all_ent[-1]:,} "
          f"(+{all_ent[-1] - all_ent[0]:,})")
    bytes_per = EMBEDDING_DIM * 4 * 3  # Adam: weight + grad + m + v = 4×
    est_table = all_ent[-1] * bytes_per / (1024**2)
    print(f"  Est table: {est_table:.0f} MB ({all_ent[-1]:,} × {bytes_per}B/entry)")
    print()

    # Total time
    print(f"Total time: {total_time:.1f}s ({total_time/epochs:.1f}s/epoch, "
          f"{total_time/n_batches/epochs*1000:.1f}ms/batch)")
    print()

    # NaN check
    sd = model.emb.state_dict()
    w = sd["weight"]
    nan_count = torch.isnan(w).sum().item()
    inf_count = torch.isinf(w).sum().item()
    if nan_count > 0 or inf_count > 0:
        print(f"✗ FOUND: {nan_count} NaN, {inf_count} Inf in weights!")
    else:
        print(f"✓ No NaN/Inf in weights "
              f"(min={w.min().item():+.4f}, max={w.max().item():+.4f}, "
              f"mean={w.mean().item():+.6f}, std={w.std().item():+.4f})")

    # Convergence check
    if best_auc > 0.75:
        print(f"✓ AUC > 0.75 — model is learning the signal")
    elif best_auc > 0.65:
        print(f"⚠ AUC = {best_auc:.4f} — moderate, may need more steps or tuning")
    else:
        print(f"✗ AUC = {best_auc:.4f} — model may not be learning (check data/debug)")

    # Capacity check
    if all_ent[-1] > capacity * 0.8:
        print(f"⚠ Entries ({all_ent[-1]:,}) near capacity ({capacity:,}) — "
              f"consider increasing")

    print("=" * 70)


if __name__ == "__main__":
    main()
