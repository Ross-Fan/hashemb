#!/usr/bin/env python3
"""
HashEmb stability / stress test with real TFRecord data.

Reads TFRecord (gzip compressed), parses features using the same schema
as the TF training pipeline, and runs sustained HashEmb training with
per-epoch memory/timing/AUC monitoring.

Feature schema (matching ModelConfig):
  - 63 discrete features: dis_00 … dis_62  (scalar int64)
  -  2 seq1 features:     dis_63, dis_64   (var-length, pad to 5)
  -  6 seq2 features:     dis_65–dis_70    (var-length, pad to 10 or 30)
  - Max 193 feat IDs per sample (63 discrete + 2×5 + 3×30 + 3×10)
  - Label: click (adjusted by play_score threshold 0.1)

Dependencies:
    pip install tensorflow  # for TFRecord parsing (already in your env)

Usage:
    # Default: read up to 2M records, train 100 epochs
    python examples/benchmark_real_data.py --data "/path/to/*.tfrecord.gz"

    # Long stability test
    python examples/benchmark_real_data.py --data "data/*.tfrecord.gz" --steps 500

    # Time-based: run for 30 minutes
    python examples/benchmark_real_data.py --data "data/*.tfrecord.gz" --duration 1800

    # Read ALL records (0 = unlimited)
    python examples/benchmark_real_data.py --data "data/*.tfrecord.gz" --max-records 0

    # More capacity, larger block size
    python examples/benchmark_real_data.py --data "data/*.tfrecord.gz" --capacity 20000000 --block-size 1000000
"""

import argparse
import resource
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

from hashemb import HashEmbedding

# =============================================================================
# TFRecord parsing (uses tensorflow — already in user's environment)
# =============================================================================
try:
    import tensorflow as tf
except ImportError:
    print("Error: tensorflow is required for TFRecord parsing.")
    print("  pip install tensorflow")
    sys.exit(1)

# =============================================================================
# Feature schema — matching the user's ModelConfig
# =============================================================================
def _discrete_keys():
    return [f"dis_{i:02d}" for i in range(63)]  # dis_00 … dis_62

def _seq1_keys():
    return {"dis_63": 5, "dis_64": 5}

def _seq2_keys():
    return {
        "dis_65": 30, "dis_66": 30, "dis_67": 30,
        "dis_68": 10, "dis_69": 10, "dis_70": 10,
    }

def _all_seq_keys():
    return {**_seq1_keys(), **_seq2_keys()}

DISCRETE_KEYS   = _discrete_keys()                         # 63 keys
SEQ1_KEYS       = _seq1_keys()                             # 2 keys, max 5 each
SEQ2_KEYS       = _seq2_keys()                             # 6 keys, max 10/30 each
ALL_SEQ_KEYS    = _all_seq_keys()                          # 8 keys total
N_DISCRETE      = len(DISCRETE_KEYS)                       # 63
N_SEQ           = len(ALL_SEQ_KEYS)                        # 8
MAX_FEATS       = N_DISCRETE + sum(ALL_SEQ_KEYS.values())  # 63 + 130 = 193

# =============================================================================
# Defaults (can override via CLI)
# =============================================================================
EMBEDDING_DIM   = 16
BATCH_SIZE      = 4096
MAX_RECORDS     = 2_000_000
EPOCHS          = 100
LR              = 0.01
HASH_CAPACITY   = 10_000_000
BLOCK_SIZE      = 1_000_000
SEED            = 42

PAD_VALUE = -1  # sentinel for seq padding in PyTorch tensors


# =============================================================================
# Shared TFRecord parsing (used by both pre-load and streaming paths)
# =============================================================================
def parse_single_tfrecord(example_proto):
    """Parse a single TFRecord example protobuf into (feat_ids, label) tensors.

    Mirrors the training pipeline's parse_tfrecord().
    Returns (all_ids, click) where all_ids is (MAX_FEATS,) int64.
    """
    feat_spec = {
        "click":      tf.io.FixedLenFeature([], tf.int64),
        "play_score": tf.io.FixedLenFeature([], tf.float32),
    }
    for k in DISCRETE_KEYS:
        feat_spec[k] = tf.io.FixedLenFeature([], tf.int64)
    for k in ALL_SEQ_KEYS:
        feat_spec[k] = tf.io.VarLenFeature(tf.int64)

    parsed = tf.io.parse_single_example(example_proto, feat_spec)

    # Discrete features → stack into 1D tensor
    disc = tf.stack([parsed[k] for k in DISCRETE_KEYS])  # (63,)

    # Sequence features → dense pad, flatten
    seqs = []
    for k, max_len in ALL_SEQ_KEYS.items():
        dense = tf.sparse.to_dense(parsed[k], default_value=PAD_VALUE)
        cur_len = tf.shape(dense)[0]
        if cur_len < max_len:
            dense = tf.pad(dense, [[0, max_len - cur_len]], constant_values=PAD_VALUE)
        else:
            dense = dense[:max_len]
        seqs.append(dense)

    # Concatenate all feature IDs: discrete + seq1 + seq2
    all_ids = tf.concat([disc] + seqs, axis=0)  # (MAX_FEATS,)

    # Label: click adjusted by play_score threshold
    click = tf.where(
        parsed["play_score"] > 0.1,
        tf.ones_like(parsed["click"]),
        parsed["click"],
    )

    return all_ids, click


# =============================================================================
# Streaming TFRecord Dataset (IterableDataset for large data)
# =============================================================================
class StreamingTFRecordDataset(torch.utils.data.IterableDataset):
    """Lazy IterableDataset that streams TFRecord files from disk.

    Designed for datasets too large to fit in memory (e.g. 100M+ records).
    Builds a fresh tf.data pipeline inside __iter__() for fork safety.

    Train/val split is at the file level: the first ``val_split`` fraction
    of sorted files is reserved for validation.
    """

    def __init__(self, file_patterns, max_records=0, split="train",
                 val_split=0.1, seed=42, shuffle_buffer_size=10000):
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")

        files = []
        for pattern in file_patterns:
            matched = tf.io.gfile.glob(pattern)
            files.extend(matched)
        if not files:
            raise FileNotFoundError(
                f"No TFRecord files found for patterns: {file_patterns}")
        files.sort()

        # File-level train/val split
        n_val_files = max(1, int(len(files) * val_split))
        if n_val_files >= len(files):
            n_val_files = max(1, len(files) - 1)  # ensure at least 1 train file
        if split == "train":
            self._files = files[n_val_files:]
        else:
            self._files = files[:n_val_files]

        self._max_records = max_records
        self._seed = seed
        self._shuffle_buffer_size = shuffle_buffer_size
        self._epoch = 0
        self.n_files = len(self._files)

    def set_epoch(self, epoch):
        """Set current epoch (controls shuffle seed). Call before each epoch."""
        self._epoch = epoch

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            my_files = list(self._files)
        else:
            my_files = self._files[worker_info.id::worker_info.num_workers]

        if not my_files:
            return iter([])

        # Shuffle file order for this epoch
        rng = np.random.RandomState(self._seed + self._epoch)
        rng.shuffle(my_files)

        # Build tf.data pipeline (fresh each __iter__, fork-safe)
        ds = tf.data.TFRecordDataset(my_files, compression_type="GZIP")
        ds = ds.map(parse_single_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)

        if self._shuffle_buffer_size > 0:
            ds = ds.shuffle(self._shuffle_buffer_size,
                            seed=self._seed + self._epoch)

        if self._max_records > 0:
            ds = ds.take(self._max_records)

        for ids_tf, lbl_tf in ds:
            ids_np = ids_tf.numpy().astype(np.int64)
            lbl_np = lbl_tf.numpy().astype(np.float32)

            # Discard all-pad samples (same as pre-load path)
            if not (ids_np != PAD_VALUE).any():
                continue

            yield (torch.from_numpy(ids_np).long(),
                   torch.tensor(float(lbl_np), dtype=torch.float32))


# =============================================================================
# Stats estimation for streaming mode
# =============================================================================
def estimate_streaming_stats(file_patterns, n_samples=10000):
    """Estimate dataset statistics from the first N records.

    Provides approximate stats without loading the full dataset.
    Returns None if n_samples <= 0.
    """
    if n_samples <= 0:
        return None

    files = []
    for pattern in file_patterns:
        matched = tf.io.gfile.glob(pattern)
        files.extend(matched)
    if not files:
        return None
    files.sort()

    ds = (tf.data.TFRecordDataset(files, compression_type="GZIP")
          .map(parse_single_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
          .take(n_samples))

    total_valid = 0
    valid_counts = []
    feat_min = float("inf")
    feat_max = float("-inf")
    n_pos = 0
    count = 0
    t0 = time.time()

    for ids_tf, lbl_tf in ds:
        ids_np = ids_tf.numpy()
        lbl_np = lbl_tf.numpy()

        valid = ids_np[ids_np != PAD_VALUE]
        n_valid = len(valid)
        if n_valid == 0:
            continue

        total_valid += n_valid
        valid_counts.append(n_valid)
        feat_min = min(feat_min, valid.min())
        feat_max = max(feat_max, valid.max())
        n_pos += int(lbl_np > 0.5)
        count += 1

    t1 = time.time()

    if count == 0:
        return None

    return {
        "n_valid_feats_mean": total_valid / count,
        "n_valid_feats_std": float(np.std(valid_counts)),
        "feat_id_min": int(feat_min),
        "feat_id_max": int(feat_max),
        "label_pos_ratio": n_pos / count,
        "n_samples_est": count,
        "est_time_s": t1 - t0,
    }


# =============================================================================
# TFRecord → numpy arrays (pre-loaded for fast PyTorch iteration)
# =============================================================================
def load_tfrecord_data(file_patterns, max_records):
    """
    Parse TFRecord files matching *file_patterns* (supports glob),
    return (feat_ids_2d, labels, n_discarded, n_valid_feats_stats).

    feat_ids_2d: (N, MAX_FEATS) int64 array,  padded with PAD_VALUE.
    labels:      (N,) float32 array.
    """
    files = []
    for pattern in file_patterns:
        matched = tf.io.gfile.glob(pattern)
        if not matched:
            print(f"  WARNING: no files match '{pattern}'")
            continue
        files.extend(matched)

    if not files:
        raise FileNotFoundError(
            f"No TFRecord files found for: {file_patterns}")

    print(f"  Found {len(files)} file(s)")
    files.sort()
    for f in files[:5]:
        print(f"    {f}")
    if len(files) > 5:
        print(f"    ... and {len(files) - 5} more")

    # ── Build tf.data pipeline ──
    raw_ds = tf.data.TFRecordDataset(files, compression_type="GZIP")
    ds = raw_ds.map(parse_single_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)

    if max_records > 0:
        ds = ds.take(max_records)

    # ── Iterate TF dataset → accumulate numpy arrays ──
    feat_rows = []
    label_rows = []
    total_feat_ids = 0
    valid_feat_count = []
    n_discarded = 0
    t0 = time.time()

    print(f"  Parsing TFRecord (max {max_records:,} records)...")
    for i, (ids_t, lbl_t) in enumerate(ds):
        ids_np = ids_t.numpy().astype(np.int64)
        lbl_np = lbl_t.numpy().astype(np.float32)

        # Quick sanity: discard samples with no valid features
        n_valid = int((ids_np != PAD_VALUE).sum())
        if n_valid == 0:
            n_discarded += 1
            continue

        feat_rows.append(ids_np)
        label_rows.append(lbl_np)
        total_feat_ids += n_valid
        valid_feat_count.append(n_valid)

        if (i + 1) % 200_000 == 0:
            elapsed = time.time() - t0
            print(f"    ... {i + 1:>10,} records ({elapsed:.1f}s, "
                  f"{len(feat_rows) / elapsed:.0f} rec/s)")

    t1 = time.time()
    n = len(feat_rows)

    if n == 0:
        raise RuntimeError("No valid records parsed — check data path and schema")

    # Stack into 2D array
    feat_arr = np.stack(feat_rows, axis=0)  # (N, MAX_FEATS)
    label_arr = np.array(label_rows, dtype=np.float32)

    avg_feats = total_feat_ids / n if n > 0 else 0
    label_pos = label_arr.sum()

    print(f"  Parsed {n:,} records in {t1 - t0:.1f}s "
          f"({n / (t1 - t0):.0f} rec/s)")
    print(f"  Discarded: {n_discarded} (all-feat-pad samples)")
    print(f"  Avg feat IDs per sample: {avg_feats:.1f} / {MAX_FEATS} max")
    print(f"  Unique feat IDs seen:    estimating during training...")

    stats = {
        "n_valid_feats_mean": avg_feats,
        "n_valid_feats_std": float(np.std(valid_feat_count)),
        "feat_id_min": int(feat_arr[feat_arr != PAD_VALUE].min()),
        "feat_id_max": int(feat_arr[feat_arr != PAD_VALUE].max()),
    }

    return feat_arr, label_arr, n_discarded, stats


# =============================================================================
# PyTorch Dataset
# =============================================================================
class RealDataDataset(Dataset):
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
# Model — simple HashEmb pool + linear
# =============================================================================
class RealDataModel(torch.nn.Module):
    """
    All feature IDs → HashEmbedding → mask out pads → mean pool → Linear → logit.

    The model is intentionally simple: the goal is to stress-test HashEmb's
    lookup + gradient + Adam step paths with real feature distributions,
    not to maximize predictive accuracy.
    """
    def __init__(self, emb_dim, capacity, lr, block_size):
        super().__init__()
        self.emb = HashEmbedding(
            emb_dim, capacity,
            optimizer="adam", lr=lr,
            initial_scale=0.01,
            block_size=block_size,
        )
        self.predict = torch.nn.Linear(emb_dim, 1)
        torch.nn.init.xavier_uniform_(self.predict.weight)
        torch.nn.init.zeros_(self.predict.bias)

    def forward(self, feat_ids):
        """
        feat_ids: (B, max_feats)  with PAD_VALUE=-1 for padding

        Mask out padded positions before pooling so they don't
        contribute to the prediction.
        """
        mask = (feat_ids != PAD_VALUE).float()             # (B, max_feats)
        valid_count = mask.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1)

        embs = self.emb(feat_ids)                          # (B, max_feats, D)
        embs = embs * mask.unsqueeze(-1)                   # zero out pads
        pooled = embs.sum(dim=1) / valid_count             # (B, D)
        return self.predict(pooled).squeeze(-1)

    def step(self):
        self.emb.step()


# =============================================================================
# Helpers
# =============================================================================
def mem_rss_mb():
    """Current RSS in MB (Linux: ru_maxrss reports KB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="HashEmb stability test with real TFRecord data")
    parser.add_argument("--data", nargs="+", required=True,
                        help="TFRecord file pattern(s), e.g. 'data/*.tfrecord.gz'")
    parser.add_argument("--max-records", type=int, default=MAX_RECORDS,
                        help="Max records to load (0 = unlimited)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--steps", type=int, default=EPOCHS,
                        help="Number of epochs (ignored if --duration set)")
    parser.add_argument("--duration", type=int, default=0,
                        help="Run for N seconds (overrides --steps)")
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--capacity", type=int, default=HASH_CAPACITY)
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Fraction of data held out for validation")
    parser.add_argument("--stream", action="store_true", default=True,
                        help="Stream data from disk (default, avoids OOM on large data)")
    parser.add_argument("--no-stream", action="store_false", dest="stream",
                        help="Pre-load all data into RAM (original behavior)")
    parser.add_argument("--stats-samples", type=int, default=10000,
                        help="Records to sample for stats estimation in stream mode")
    parser.add_argument("--shuffle-buffer", type=int, default=10000,
                        help="tf.data shuffle buffer size for streaming mode")
    parser.add_argument("--log-interval", type=int, default=10,
                        help="Print progress every N batches within each epoch")
    args = parser.parse_args()

    batch_size   = args.batch_size
    max_records  = args.max_records
    epochs       = args.steps if args.duration <= 0 else 999999
    duration     = args.duration
    lr           = args.lr
    capacity     = args.capacity
    block_size   = args.block_size
    val_split    = args.val_split

    # =========================================================================
    # Header
    # =========================================================================
    bytes_per_entry = EMBEDDING_DIM * 4 * 4  # Adam: weight + grad + m + v
    print("=" * 70)
    print("HashEmb Stability Test — Real TFRecord Data")
    print("=" * 70)
    print(f"  Data pattern:          {args.data}")
    print(f"  Mode:                  "
          f"{'STREAMING (lazy from disk)' if args.stream else 'PRE-LOAD (all in RAM)'}")
    print(f"  Max records:           {max_records:,} (0=unlimited)")
    print(f"  Batch size:            {batch_size:,}")
    print(f"  Feat IDs / sample:     ≤ {MAX_FEATS}  "
          f"({N_DISCRETE} discrete + {N_SEQ} seq)")
    print(f"  Embedding dim:         {EMBEDDING_DIM}")
    print(f"  Hash capacity:         {capacity:,}")
    print(f"  Block size:            {block_size:,}")
    print(f"  Optimizer:             Adam, lr={lr}")
    print(f"  Per-entry memory:      {bytes_per_entry} B (4 × float32)")
    print(f"  Epochs:                {epochs if not duration else 'until timeout'}")
    if duration:
        print(f"  Duration limit:        {duration}s")
    print(f"  Val split:             {val_split:.0%}")
    print(f"  Debug:                 {args.debug}")
    print()

    # =========================================================================
    # Load data (streaming or pre-load)
    # =========================================================================
    mem0 = mem_rss_mb()
    print(f"[MEM] Before load: {mem0:.0f} MB")

    n_discarded = 0
    ds_stats = {}
    lookups_per_batch = batch_size * MAX_FEATS

    if args.stream:
        # ── Streaming mode ──
        print(f"  Mode: STREAMING (lazy read from disk,"
              f" shuffle_buffer={args.shuffle_buffer:,})")

        # Estimate stats from first K records
        t_load = time.time()
        ds_stats = estimate_streaming_stats(args.data, args.stats_samples)
        t_load = time.time() - t_load

        if ds_stats is not None:
            print(f"  Stats estimated from {ds_stats['n_samples_est']:,} records"
                  f" ({ds_stats.get('est_time_s', 0):.1f}s):")
            print(f"    Est. avg valid feats/sample: {ds_stats['n_valid_feats_mean']:.1f}")
            print(f"    Est. feat_id range: ["
                  f"{ds_stats['feat_id_min']:,}, {ds_stats['feat_id_max']:,}]")
            print(f"    Est. label pos ratio: {ds_stats['label_pos_ratio']:.3f}")
        else:
            print("  Stats estimation skipped (--stats-samples=0)")

        train_ds = StreamingTFRecordDataset(
            args.data,
            max_records=max_records,
            split="train",
            val_split=val_split,
            seed=SEED,
            shuffle_buffer_size=args.shuffle_buffer,
        )
        val_ds = StreamingTFRecordDataset(
            args.data,
            max_records=max_records,
            split="val",
            val_split=val_split,
            seed=SEED,
            shuffle_buffer_size=0,  # no shuffle for val
        )

        print(f"  Train files: {train_ds.n_files}  Val files: {val_ds.n_files}")

        train_loader = DataLoader(
            train_ds, batch_size=batch_size,
            collate_fn=collate_fn, drop_last=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size,
            collate_fn=collate_fn, drop_last=False,
        )

        # n_batches discovered during first epoch
        n_batches = None

        mem1 = mem_rss_mb()
        if ds_stats:
            print(f"  Est. label balance: positive ratio = "
                  f"{ds_stats['label_pos_ratio']:.1%}")
        print(f"[MEM] After dataset init:  {mem1:.0f} MB  "
              f"(+{mem1 - mem0:.0f} MB)")
        print()

    else:
        # ── Pre-load mode (original behavior) ──
        print("  Mode: PRE-LOAD (all data in RAM)")

        t_load = time.time()
        feat_arr, label_arr, n_discarded, ds_stats = \
            load_tfrecord_data(args.data, max_records)
        t_load = time.time() - t_load

        mem1 = mem_rss_mb()
        n_total = len(label_arr)
        n_pos = int(label_arr.sum())
        print(f"  Label balance: {n_pos:,} / {n_total:,} "
              f"({100 * n_pos / n_total:.1f}%)")
        print(f"  Raw feat ID range: [{ds_stats['feat_id_min']:,}, "
              f"{ds_stats['feat_id_max']:,}]")
        print(f"  Avg valid feats/sample: {ds_stats['n_valid_feats_mean']:.1f} "
              f"± {ds_stats['n_valid_feats_std']:.1f}")
        print(f"[MEM] After load:  {mem1:.0f} MB  (+{mem1 - mem0:.0f} MB, "
              f"{t_load:.1f}s)")
        print()

        # Train / val split
        rng = np.random.RandomState(SEED)
        n_val = max(1, int(n_total * val_split))
        perm = rng.permutation(n_total)
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

        train_ds_inner = RealDataDataset(feat_arr[train_idx], label_arr[train_idx])
        val_ds_inner   = RealDataDataset(feat_arr[val_idx],   label_arr[val_idx])

        n_batches = len(train_ds_inner) // batch_size

        train_loader = DataLoader(train_ds_inner, batch_size, shuffle=True,
                                  collate_fn=collate_fn, drop_last=True)
        val_loader   = DataLoader(val_ds_inner,   batch_size, shuffle=False,
                                  collate_fn=collate_fn, drop_last=False)

        print(f"  Train: {len(train_ds_inner):,}  Val: {len(val_ds_inner):,}  "
              f"Batches/epoch: {n_batches}")
        print()

    # =========================================================================
    # Model
    # =========================================================================
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    mem2 = mem_rss_mb()
    print(f"[MEM] Before model: {mem2:.0f} MB")

    model = RealDataModel(EMBEDDING_DIM, capacity, lr, block_size)
    opt = torch.optim.Adam(model.predict.parameters(), lr=lr)

    mem3 = mem_rss_mb()
    print(f"  Initial entries: {model.emb.num_entries:,}")
    print(f"[MEM] After model:  {mem3:.0f} MB  (+{mem3 - mem2:.0f} MB)")
    print()

    # =========================================================================
    # Training with per-epoch monitoring
    # =========================================================================
    snap_mem    = []   # (epoch, rss_mb, num_entries)
    snap_timing = []   # (epoch, fwd_ms, bwd_ms, step_ms, total_ms)
    snap_auc    = []   # (epoch, auc)
    prev_mem    = mem3
    prev_ent    = 0
    wall_start  = time.time()

    print("-" * 95)
    print(f"{'Ep':>4s} | {'loss':>7s} {'auc':>7s} | "
          f"{'entries':>10s} {'RSS(MB)':>8s} {'ΔMB':>6s} | "
          f"{'fwd':>6s} {'bwd':>6s} {'step':>6s} {'total':>6s} | "
          f"{'Mlookup/s':>9s}")
    print("-" * 95)

    epoch = 0
    while True:
        epoch += 1
        if epoch > epochs:
            break
        if duration and (time.time() - wall_start) > duration:
            break

        # Set epoch for deterministic per-epoch shuffle in streaming mode
        if args.stream:
            train_ds.set_epoch(epoch)

        model.train()
        ep_loss = 0.0
        ep_fwd, ep_bwd, ep_step, ep_tot = [], [], [], []
        n_batches_this_epoch = 0

        # Per-interval accumulators (reset every --log-interval batches)
        int_loss = 0.0
        int_fwd, int_bwd, int_step, int_tot = [], [], [], []
        int_data_time = []                     # data-loading time per batch
        int_scores, int_labels = [], []        # for per-interval training AUC
        int_t_start = time.perf_counter()
        t_prev_end = time.perf_counter()       # for measuring data-load time
        log_interval = args.log_interval

        for bi, (batch_ids, labels) in enumerate(train_loader):
            # ── Data load time: wall clock since end of previous batch ──
            t_fetch = time.perf_counter()
            dt_data = t_fetch - t_prev_end

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
            t_prev_end = t3

            dt_fwd, dt_bwd, dt_step = t1 - t0, t2 - t1, t3 - t2
            dt_comp = dt_fwd + dt_bwd + dt_step
            dt_total = dt_data + dt_comp

            ep_loss += loss.item()
            ep_fwd.append(dt_fwd)
            ep_bwd.append(dt_bwd)
            ep_step.append(dt_step)
            ep_tot.append(dt_comp)
            n_batches_this_epoch += 1

            int_loss += loss.item()
            int_fwd.append(dt_fwd)
            int_bwd.append(dt_bwd)
            int_step.append(dt_step)
            int_tot.append(dt_comp)
            int_data_time.append(dt_data)
            int_scores.append(torch.sigmoid(logits).detach().cpu().numpy())
            int_labels.append(labels.detach().cpu().numpy())

            # ── Slow batch detection ──
            if dt_total > 5.0:
                cur_ent = model.emb.num_entries
                print(f"  ⚠ [Ep {epoch}, bat {bi + 1}] SLOW: "
                      f"data={dt_data:.2f}s comp={dt_comp:.3f}s "
                      f"(fwd={dt_fwd:.3f} bwd={dt_bwd:.3f} step={dt_step:.3f})  "
                      f"ent={cur_ent:,}",
                      flush=True)

            # ── Per-interval progress ──
            if (bi + 1) % log_interval == 0:
                n_int = len(int_fwd)
                int_elapsed = time.perf_counter() - int_t_start
                avg_per_bat  = int_elapsed / n_int * 1000  # ms/batch
                avg_i_loss   = int_loss / n_int
                avg_i_fwd    = np.mean(int_fwd) * 1000
                avg_i_bwd    = np.mean(int_bwd) * 1000
                avg_i_step   = np.mean(int_step) * 1000
                avg_data     = np.mean(int_data_time) * 1000
                avg_comp     = np.mean(int_tot) * 1000
                i_tput       = lookups_per_batch / (avg_comp / 1000) / 1e6
                n_ent_now   = model.emb.num_entries
                batch_info   = f"{bi + 1}"
                if n_batches is not None:
                    batch_info += f"/{n_batches}"

                # Per-interval training AUC (from this interval's batches)
                try:
                    i_auc = roc_auc_score(
                        np.concatenate(int_labels),
                        np.concatenate(int_scores))
                except ValueError:
                    i_auc = float("nan")

                print(f"  [Ep {epoch}, bat {batch_info:<9s}] "
                      f"loss={avg_i_loss:.4f} auc={i_auc:.4f}  "
                      f"ent={n_ent_now:>9,d}  "
                      f"data={avg_data:.0f}ms comp={avg_comp:.0f}ms("
                      f"fwd={avg_i_fwd:.0f} bwd={avg_i_bwd:.0f} step={avg_i_step:.0f})  "
                      f"bat={avg_per_bat:.0f}ms  tput={i_tput:.1f}M/s",
                      flush=True)

                int_loss = 0.0
                int_fwd, int_bwd, int_step, int_tot = [], [], [], []
                int_data_time = []
                int_scores, int_labels = [], []
                int_t_start = time.perf_counter()

        # ── Validation ──
        model.eval()
        all_scores, all_labels_gt = [], []
        with torch.no_grad():
            for batch_ids, labels in val_loader:
                logits = model(batch_ids)
                all_scores.append(torch.sigmoid(logits).numpy())
                all_labels_gt.append(labels.numpy())

        try:
            auc = roc_auc_score(
                np.concatenate(all_labels_gt),
                np.concatenate(all_scores))
        except ValueError:
            auc = 0.5  # single-class batch edge case

        # ── Metrics ──
        n_ent      = model.emb.num_entries
        cur_mem    = mem_rss_mb()
        mem_delta  = cur_mem - prev_mem
        ent_delta  = n_ent - prev_ent

        avg_fwd  = np.mean(ep_fwd) * 1000
        avg_bwd  = np.mean(ep_bwd) * 1000
        avg_step = np.mean(ep_step) * 1000
        avg_tot  = np.mean(ep_tot) * 1000
        tput     = lookups_per_batch / (avg_tot / 1000) / 1e6

        # Discover n_batches from first streaming epoch
        if n_batches is None:
            n_batches = n_batches_this_epoch
            if args.stream:
                print(f"  Discovered {n_batches} batches/epoch from first epoch")

        snap_mem.append((epoch, cur_mem, n_ent))
        snap_timing.append((epoch, avg_fwd, avg_bwd, avg_step, avg_tot))
        snap_auc.append((epoch, auc))

        avg_loss = ep_loss / max(n_batches_this_epoch, 1)
        print(f"{epoch:4d} | {avg_loss:7.4f} {auc:7.4f} | "
              f"{n_ent:10,d} {cur_mem:8.0f} {mem_delta:+6.0f} | "
              f"{avg_fwd:6.1f} {avg_bwd:6.1f} {avg_step:6.1f} {avg_tot:6.1f} | "
              f"{tput:9.1f}")

        # ── Extended debug every 10 epochs ──
        if args.debug and epoch % 10 == 0:
            sd = model.emb.state_dict()
            w = sd["weight"]
            print(f"  [DEBUG ep{epoch}] "
                  f"ent+{ent_delta:,}  "
                  f"w=[{w.min().item():+.4f}, {w.max().item():+.4f}]  "
                  f"blocks≈{n_ent // block_size + 1}  "
                  f"mb/kent={mem_delta / (ent_delta + 1) * 1000:.2f}")

        prev_mem = cur_mem
        prev_ent = n_ent

    total_time = time.time() - wall_start

    # =========================================================================
    # Summary
    # =========================================================================
    print("-" * 95)
    print()
    print("=" * 70)
    print("Stability Summary")
    print("=" * 70)
    print(f"  Total epochs:   {epoch}")
    print(f"  Total time:     {total_time:.1f}s "
          f"({total_time / max(epoch, 1):.1f}s/epoch)")
    print(f"  Total batches:  {epoch * (n_batches or 0):,}")
    print(f"  Total lookups:  {epoch * (n_batches or 0) * batch_size * MAX_FEATS:,}")
    print()

    # ── Entry / memory growth ──
    ent_growth = mem_growth = mb_per_kent = 0.0
    if len(snap_mem) >= 2:
        ent_first = snap_mem[0][2]
        ent_last  = snap_mem[-1][2]
        mem_first = snap_mem[0][1]
        mem_last  = snap_mem[-1][1]
        ent_growth = ent_last - ent_first
        mem_growth = mem_last - mem_first
        mb_per_kent = (mem_growth / max(ent_growth, 1)) * 1000

        print("Entry & Memory Growth:")
        print(f"  Start entries:  {ent_first:>10,}")
        print(f"  Final entries:  {ent_last:>10,}  (+{ent_growth:,})")
        print(f"  Start RSS:      {mem_first:>10.0f} MB")
        print(f"  Final RSS:      {mem_last:>10.0f} MB  (+{mem_growth:.0f} MB)")
        print(f"  MB / k entries: {mb_per_kent:.2f}  "
              f"(theoretical: {bytes_per_entry * 1000 / (1024**2):.2f} "
              f"MB/k for Adam 4× buffers)")
        print()

    # ── Timing stability ──
    if len(snap_timing) >= 4:
        print("Timing Stability (first 3 vs last 3 epochs):")
        for idx, label in [(1, "fwd"), (2, "bwd"), (3, "step"), (4, "total")]:
            first3 = np.mean([s[idx] for s in snap_timing[:3]])
            last3  = np.mean([s[idx] for s in snap_timing[-3:]])
            delta  = last3 - first3
            delta_pct = 100 * delta / (first3 + 0.001)
            flag = ""
            if delta_pct > 5:
                flag = f"  ⚠ +{delta_pct:.0f}% slower"
            elif delta_pct < -5:
                flag = "  (faster)"
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
    nan_count  = torch.isnan(w).sum().item()
    inf_count  = torch.isinf(w).sum().item()
    w_min, w_max = w.min().item(), w.max().item()
    w_mean, w_std = w.mean().item(), w.std().item()

    if nan_count > 0 or inf_count > 0:
        print(f"✗ WEIGHT ISSUE: {nan_count} NaN, {inf_count} Inf detected!")
    else:
        print(f"✓ Weights OK:  min={w_min:+.4f}  max={w_max:+.4f}  "
              f"mean={w_mean:+.6f}  std={w_std:+.4f}")

    # ── Verdict ──
    print()
    issues = []
    if nan_count > 0 or inf_count > 0:
        issues.append("NaN/Inf in weights")
    if ent_growth <= 0 and epoch > 1:
        issues.append("No entry growth after first epoch — "
                      "is --max-records large enough?")
    theoretical_mb_per_k = bytes_per_entry * 1000 / (1024 ** 2)
    if mb_per_kent > theoretical_mb_per_k * 2:
        issues.append(f"memory/entry {mb_per_kent:.1f} MB/k "
                      f"vs theoretical {theoretical_mb_per_k:.1f}")
    if len(snap_timing) >= 4:
        first3_total = np.mean([s[4] for s in snap_timing[:3]])
        last3_total  = np.mean([s[4] for s in snap_timing[-3:]])
        slowdown_pct = 100 * (last3_total - first3_total) / (first3_total + 0.001)
        if slowdown_pct > 20:
            issues.append(f"throughput degraded {slowdown_pct:.0f}%")
        elif slowdown_pct > 10:
            issues.append(f"throughput degraded {slowdown_pct:.0f}% (moderate)")

    if not issues:
        print("✓ STABILITY PASS — no issues detected")
    else:
        for issue in issues:
            print(f"⚠ {issue}")
    print("=" * 70)


if __name__ == "__main__":
    main()
