#!/usr/bin/env python3
"""
HashEmb stability / stress test with real data (Parquet format).

Reads Parquet files (snappy compressed), reconstructs feat_ids from
columnar storage, and runs sustained HashEmb training with per-epoch
memory/timing/AUC monitoring.

Feature schema (matching Spark V005 → Parquet):
  - 63 discrete features: dis_00 … dis_62  (scalar int64)
  - 12 sequence features: dis_63 … dis_74  (list<int>, pad to fixed max lengths)
  - Max 233 feat IDs per sample (63 + 5+5+30+30+30+10+10+10+10+10+10+10)
  - Label: click (adjusted by play_score threshold 0.1)

Dependencies:
    pip install pyarrow  # for Parquet reading

Usage:
    # Default: read up to 2M records, 1 epoch (single pass through data)
    python examples/benchmark_real_data.py --data "/path/to/*.parquet"

    # Long stability test
    python examples/benchmark_real_data.py --data "data/*.parquet" --steps 500

    # Read ALL records (0 = unlimited)
    python examples/benchmark_real_data.py --data "data/*.parquet" --max-records 0

    # More capacity, larger block size
    python examples/benchmark_real_data.py --data "data/*.parquet" --capacity 20000000 --block-size 1000000
"""

import argparse
import glob
import os
import resource
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from hashemb import HashEmbedding

# =============================================================================
# Parquet reading (pyarrow — lightweight, no TensorFlow dependency)
# =============================================================================
try:
    import pyarrow.parquet as pq
except ImportError:
    print("Error: pyarrow is required for Parquet reading.")
    print("  pip install pyarrow")
    sys.exit(1)

# =============================================================================
# Feature schema — matching Spark V005 → Parquet columns
# =============================================================================
# 63 discrete (scalar int) features: dis_00 … dis_62
N_DISCRETE = 63

# 12 sequence (list<int>) features with their max lengths after padding
SEQ_MAX_LENS = {
    "dis_63": 5,   # ibb_tags_llm
    "dis_64": 5,   # ibb_titles
    "dis_65": 30,  # app_feat
    "dis_66": 30,  # x_gameid_app_feat
    "dis_67": 30,  # user_sequence_basic
    "dis_68": 10,  # user_continue_play_sequence
    "dis_69": 10,  # user_real_click_sequence
    "dis_70": 10,  # user_real_play_sequence
    "dis_71": 10,  # user_seq_cate_stat_bucket01
    "dis_72": 10,  # user_seq_cate_stat_bucket02
    "dis_73": 10,  # user_seq_cate_stat_bucket03
    "dis_74": 10,  # x_game_id_user_cate_stat_03
}

DISCRETE_KEYS = [f"dis_{i:02d}" for i in range(N_DISCRETE)]  # dis_00 … dis_62
SEQ_KEYS      = list(SEQ_MAX_LENS.keys())                    # dis_63 … dis_74
N_SEQ         = len(SEQ_KEYS)                                # 12
MAX_FEATS     = N_DISCRETE + sum(SEQ_MAX_LENS.values())      # 63 + 170 = 233

# =============================================================================
# Defaults (can override via CLI)
# =============================================================================
EMBEDDING_DIM   = 16
BATCH_SIZE      = 4096
MAX_RECORDS     = 2_000_000
EPOCHS          = 1
LR              = 0.01
HASH_CAPACITY   = 10_000_000
BLOCK_SIZE      = 1_000_000
SEED            = 42

PAD_VALUE = -1  # sentinel for seq padding in PyTorch tensors


# =============================================================================
# Parquet → feat_ids conversion
# =============================================================================
def build_feat_ids_from_batch(batch):
    """Convert a pyarrow RecordBatch into (feat_ids, labels) numpy arrays.

    Parameters
    ----------
    batch : pyarrow.RecordBatch
        Columns: ``dis_00`` … ``dis_74`` (int / list<int>), ``click`` (int),
        ``play_score`` (float).

    Returns
    -------
    feat_ids : (N, MAX_FEATS) int64, PAD_VALUE=-1 for padding
    labels   : (N,) float32, 1.0 if play_score > 0.1 else click value
    """
    N = batch.num_rows

    # Allocate output buffer filled with PAD_VALUE
    feat_ids = np.full((N, MAX_FEATS), PAD_VALUE, dtype=np.int64)

    # ── Discrete features: each is a scalar int64 column ──
    col_offset = 0
    for k in DISCRETE_KEYS:
        feat_ids[:, col_offset] = batch.column(k).to_numpy().astype(np.int64)
        col_offset += 1

    # ── Sequence features: each is a list<int> column → pad to max_len ──
    for k in SEQ_KEYS:
        max_len = SEQ_MAX_LENS[k]
        lists = batch.column(k).to_pylist()
        for i, row_list in enumerate(lists):
            if row_list is None or len(row_list) == 0:
                continue
            n_copy = min(len(row_list), max_len)
            feat_ids[i, col_offset:col_offset + n_copy] = row_list[:n_copy]
        col_offset += max_len

    # ── Labels: click adjusted by play_score threshold ──
    click_col = batch.column("click").to_numpy().astype(np.float32)
    play_score_col = batch.column("play_score").to_numpy().astype(np.float32)
    labels = np.where(play_score_col > 0.1, 1.0, click_col).astype(np.float32)

    return feat_ids, labels


# =============================================================================
# Streaming Parquet Dataset (IterableDataset for large data)
# =============================================================================
class StreamingParquetDataset(torch.utils.data.IterableDataset):
    """Lazy IterableDataset that streams Parquet files from disk via pyarrow.

    Designed for datasets too large to fit in memory (e.g. 100M+ records).
    Reads Parquet files in configurable row-group-sized chunks using
    ``pq.ParquetFile.iter_batches()``.
    """

    def __init__(self, file_patterns, max_records=0, seed=42,
                 parquet_batch_size=65536):
        files = []
        for pattern in file_patterns:
            matched = glob.glob(pattern)
            files.extend(matched)
        if not files:
            raise FileNotFoundError(
                f"No Parquet files found for patterns: {file_patterns}")
        files.sort()

        self._files = files
        self._max_records = max_records
        self._seed = seed
        self._parquet_batch_size = parquet_batch_size
        self._epoch = 0
        self.n_files = len(files)

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

        records_yielded = 0
        for file_path in my_files:
            pf = pq.ParquetFile(file_path)
            for batch in pf.iter_batches(batch_size=self._parquet_batch_size):
                feat_ids, labels = build_feat_ids_from_batch(batch)

                # Discard all-pad samples
                valid_mask = (feat_ids != PAD_VALUE).any(axis=1)
                feat_ids = feat_ids[valid_mask]
                labels = labels[valid_mask]

                for i in range(len(labels)):
                    yield (torch.from_numpy(feat_ids[i]).long(),
                           torch.tensor(float(labels[i]), dtype=torch.float32))
                    records_yielded += 1
                    if self._max_records > 0 and records_yielded >= self._max_records:
                        return


# =============================================================================
# Parquet stats estimation (sampling first N records)
# =============================================================================
def estimate_parquet_stats(file_patterns, n_samples=10000):
    """Estimate dataset statistics from the first ~N records.

    Reads Parquet files sequentially until n_samples valid records are seen.
    Returns None if n_samples <= 0 or no records found.
    """
    if n_samples <= 0:
        return None

    files = []
    for pattern in file_patterns:
        files.extend(glob.glob(pattern))
    if not files:
        return None
    files.sort()

    total_valid = 0
    valid_counts = []
    feat_min = float("inf")
    feat_max = float("-inf")
    n_pos = 0
    count = 0
    t0 = time.time()

    for f in files:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=min(n_samples, 10000)):
            feat_ids, labels = build_feat_ids_from_batch(batch)
            for i in range(len(labels)):
                ids = feat_ids[i]
                valid = ids[ids != PAD_VALUE]
                n_valid = len(valid)
                if n_valid == 0:
                    continue

                total_valid += n_valid
                valid_counts.append(n_valid)
                feat_min = min(feat_min, valid.min())
                feat_max = max(feat_max, valid.max())
                n_pos += int(labels[i] > 0.5)
                count += 1
                if count >= n_samples:
                    break
            if count >= n_samples:
                break
        if count >= n_samples:
            break

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
        description="HashEmb stability test with real Parquet data")
    parser.add_argument("--data", nargs="+", required=True,
                        help="Parquet file pattern(s), e.g. 'data/*.parquet'")
    parser.add_argument("--val-data", nargs="+", default=None,
                        help="Separate validation Parquet file pattern(s) "
                             "(default: no validation)")
    parser.add_argument("--max-records", type=int, default=MAX_RECORDS,
                        help="Max records to load per epoch (0 = unlimited)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--steps", type=int, default=EPOCHS,
                        help="Number of epochs (ignored if --duration set)")
    parser.add_argument("--duration", type=int, default=0,
                        help="Run for N seconds (overrides --steps)")
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--capacity", type=int, default=HASH_CAPACITY)
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--stats-samples", type=int, default=10000,
                        help="Records to sample for stats estimation")
    parser.add_argument("--parquet-batch-size", type=int, default=65536,
                        help="Rows per parquet.iter_batches() chunk")
    parser.add_argument("--log-interval", type=int, default=10,
                        help="Print progress every N batches within each epoch")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="DataLoader workers for prefetch (0=main process only)")
    parser.add_argument("--prefetch-factor", type=int, default=2,
                        help="Batches pre-loaded per worker (default: 2)")
    parser.add_argument("--save", type=str, default=None,
                        help="Save checkpoint to this path after training")
    parser.add_argument("--resume", type=str, default=None,
                        help="Load checkpoint from this path before training")
    parser.add_argument("--evict-min-count", type=int, default=0,
                        help="Evict keys with update_count < N (0=disabled)")
    parser.add_argument("--evict-max-idle-days", type=int, default=0,
                        help="Evict keys idle for > N days (0=disabled)")
    parser.add_argument("--evict-combine", type=str, default="and",
                        choices=["and", "or", ""],
                        help="Eviction combination logic (default: and)")
    args = parser.parse_args()

    batch_size   = args.batch_size
    max_records  = args.max_records
    epochs       = args.steps if args.duration <= 0 else 999999
    duration     = args.duration
    lr           = args.lr
    capacity     = args.capacity
    block_size   = args.block_size

    # =========================================================================
    # Header
    # =========================================================================
    bytes_per_entry = EMBEDDING_DIM * 4 * 4  # Adam: weight + grad + m + v
    print("=" * 70)
    print("HashEmb Stability Test — Real Parquet Data")
    print("=" * 70)
    print(f"  Data pattern:          {args.data}")
    print(f"  Max records/epoch:     {max_records:,} (0=unlimited)")
    print(f"  Batch size:            {batch_size:,}")
    print(f"  Parquet chunk size:    {args.parquet_batch_size:,}")
    print(f"  DataLoader workers:    {args.num_workers}")
    print(f"  Feat IDs / sample:     <= {MAX_FEATS}  "
          f"({N_DISCRETE} discrete + {N_SEQ} seq)")
    print(f"  Embedding dim:         {EMBEDDING_DIM}")
    print(f"  Hash capacity:         {capacity:,}")
    print(f"  Block size:            {block_size:,}")
    print(f"  Optimizer:             Adam, lr={lr}")
    print(f"  Per-entry memory:      {bytes_per_entry} B (4 x float32)")
    print(f"  Epochs:                {epochs if not duration else 'until timeout'}")
    if duration:
        print(f"  Duration limit:        {duration}s")
    if args.val_data:
        print(f"  Validation data:       {args.val_data}")
    else:
        print(f"  Validation:            SKIP (no --val-data)")
    if args.resume:
        print(f"  Resume from:           {args.resume}")
    if args.save:
        print(f"  Save to:               {args.save}")
    print(f"  Debug:                 {args.debug}")
    print()

    # =========================================================================
    # Load data (always streaming from Parquet)
    # =========================================================================
    mem0 = mem_rss_mb()
    print(f"[MEM] Before load: {mem0:.0f} MB")

    lookups_per_batch = batch_size * MAX_FEATS

    # ── Estimate stats from first K records ──
    t_est = time.time()
    ds_stats = estimate_parquet_stats(args.data, args.stats_samples)
    t_est = time.time() - t_est

    if ds_stats is not None:
        print(f"  Stats estimated from {ds_stats['n_samples_est']:,} records"
              f" ({ds_stats.get('est_time_s', 0):.1f}s):")
        print(f"    Est. avg valid feats/sample: {ds_stats['n_valid_feats_mean']:.1f}")
        print(f"    Est. feat_id range: ["
              f"{ds_stats['feat_id_min']:,}, {ds_stats['feat_id_max']:,}]")
        print(f"    Est. label pos ratio: {ds_stats['label_pos_ratio']:.3f}")
    else:
        print("  Stats estimation skipped (--stats-samples=0)")

    train_ds = StreamingParquetDataset(
        args.data,
        max_records=max_records,
        seed=SEED,
        parquet_batch_size=args.parquet_batch_size,
    )
    print(f"  Train files: {train_ds.n_files}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        collate_fn=collate_fn, drop_last=True,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )

    val_loader = None
    if args.val_data:
        val_ds = StreamingParquetDataset(
            args.val_data,
            max_records=max_records,
            seed=SEED,
            parquet_batch_size=args.parquet_batch_size,
        )
        print(f"  Val files:   {val_ds.n_files}")

        val_loader = DataLoader(
            val_ds, batch_size=batch_size,
            collate_fn=collate_fn, drop_last=False,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
        )

    mem1 = mem_rss_mb()
    if ds_stats:
        print(f"  Est. label balance: positive ratio = "
              f"{ds_stats['label_pos_ratio']:.1%}")
    print(f"[MEM] After dataset init:  {mem1:.0f} MB  "
          f"(+{mem1 - mem0:.0f} MB)")
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

    # ── Resume from checkpoint ──
    # HashEmb uses C++ binary save/load (bucket-by-bucket, zero extra memory).
    # Dense model + optimizer use torch.save/load (.pt, tens of MB).
    resume_epoch = 0
    if args.resume:
        binary_path = args.resume.replace('.pt', '.hashemb')
        if os.path.exists(binary_path):
            t_load = time.time()
            model.emb.load(binary_path)
            dt_load = time.time() - t_load
            ckpt = torch.load(args.resume, map_location="cpu", weights_only=True)
            model.predict.load_state_dict(ckpt["dense"])
            opt.load_state_dict(ckpt["opt"])
            resume_epoch = ckpt.get("epoch", 0)
            print(f"  [RESUME] Hash table from {os.path.basename(binary_path)}"
                  f"  ({dt_load:.1f}s)")
            print(f"           Dense model from {os.path.basename(args.resume)}")
            print(f"           prev_epoch={resume_epoch}  "
                  f"entries={model.emb.num_entries:,}")
        else:
            print(f"  [RESUME] {binary_path} not found, cold start")

    mem3 = mem_rss_mb()
    print(f"  Initial entries: {model.emb.num_entries:,}")
    print(f"[MEM] After model:  {mem3:.0f} MB  (+{mem3 - mem2:.0f} MB)")
    print()

    # =========================================================================
    # Training with per-epoch monitoring
    # =========================================================================
    n_batches = None   # discovered during first epoch
    snap_mem    = []   # (epoch, rss_mb, num_entries)
    snap_timing = []   # (epoch, fwd_ms, bwd_ms, step_ms, total_ms)
    snap_auc    = []   # (epoch, auc)
    prev_mem    = mem3
    prev_ent    = 0
    wall_start  = time.time()

    print("-" * 95)
    print(f"{'Ep':>4s} | {'loss':>7s} {'auc':>7s} | "
          f"{'entries':>10s} {'RSS(MB)':>8s} {'dMB':>6s} | "
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

        train_ds.set_epoch(epoch)

        model.train()
        ep_loss = 0.0
        ep_fwd, ep_bwd, ep_step, ep_tot = [], [], [], []
        n_batches_this_epoch = 0

        # Per-interval accumulators (reset every --log-interval batches)
        int_loss = 0.0
        int_fwd, int_bwd, int_step, int_tot = [], [], [], []
        int_data_time = []
        int_scores, int_labels = [], []
        int_t_start = time.perf_counter()
        t_prev_end = time.perf_counter()
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
                print(f"  [Ep {epoch}, bat {bi + 1}] SLOW: "
                      f"data={dt_data:.2f}s comp={dt_comp:.3f}s "
                      f"(fwd={dt_fwd:.3f} bwd={dt_bwd:.3f} step={dt_step:.3f})  "
                      f"ent={cur_ent:,}",
                      flush=True)

            # ── Per-interval progress ──
            if (bi + 1) % log_interval == 0:
                n_int = len(int_fwd)
                int_elapsed = time.perf_counter() - int_t_start
                avg_per_bat  = int_elapsed / n_int * 1000
                avg_i_loss   = int_loss / n_int
                avg_i_fwd    = np.mean(int_fwd) * 1000
                avg_i_bwd    = np.mean(int_bwd) * 1000
                avg_i_step   = np.mean(int_step) * 1000
                avg_data     = np.mean(int_data_time) * 1000
                avg_comp     = np.mean(int_tot) * 1000
                i_tput       = lookups_per_batch / (avg_comp / 1000) / 1e6
                n_ent_now    = model.emb.num_entries
                batch_info   = f"{bi + 1}"
                if n_batches is not None:
                    batch_info += f"/{n_batches}"

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
        if val_loader is not None:
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
                auc = 0.5
        else:
            auc = float("nan")

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

        # Discover n_batches from first epoch
        if n_batches is None:
            n_batches = n_batches_this_epoch
            print(f"  Discovered {n_batches} batches/epoch from first epoch")

        snap_mem.append((epoch, cur_mem, n_ent))
        snap_timing.append((epoch, avg_fwd, avg_bwd, avg_step, avg_tot))
        snap_auc.append((epoch, auc))

        avg_loss = ep_loss / max(n_batches_this_epoch, 1)
        auc_str = f"{auc:7.4f}" if not np.isnan(auc) else "     --"
        print(f"{epoch:4d} | {avg_loss:7.4f} {auc_str:>7s} | "
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
                  f"blocks={n_ent // block_size + 1}  "
                  f"mb/kent={mem_delta / (ent_delta + 1) * 1000:.2f}")

        prev_mem = cur_mem
        prev_ent = n_ent

    total_time = time.time() - wall_start

    # ── Save checkpoint ──
    # HashEmb → C++ binary (bucket-by-bucket, zero extra memory).
    # Dense + optimizer → torch.save (tens of MB).
    if args.save:
        binary_path = args.save.replace('.pt', '.hashemb')
        max_idle_steps = args.evict_max_idle_days * (n_batches or 0) if args.evict_max_idle_days > 0 else 0
        print(f"  [SAVE] Writing hash table to {binary_path} ...", flush=True)
        entries_before = model.emb.num_entries
        entries_written = model.emb.save(binary_path,
                       min_count=args.evict_min_count,
                       max_idle_steps=max_idle_steps,
                       combine=args.evict_combine)
        entries_evicted = entries_before - entries_written
        print(f"  [SAVE] Hash table done ({entries_written:,} entries, {entries_evicted:,} evicted)", flush=True)
        print(f"  [SAVE] Writing dense model to {args.save} ...", flush=True)
        dense_ckpt = {
            "dense": model.predict.state_dict(),
            "opt": opt.state_dict(),
            "epoch": epoch,
        }
        try:
            torch.save(dense_ckpt, args.save)
            print(f"  [SAVE] Dense model done", flush=True)
        except Exception as e:
            print(f"  [SAVE] ERROR saving dense model: {e}", flush=True)
            import traceback
            traceback.print_exc()
        print(f"\n  [SAVE] Hash table → {binary_path}")
        if args.evict_min_count > 0 or args.evict_max_idle_days > 0:
            print(f"         Eviction: min_count={args.evict_min_count}"
                  f" max_idle_steps={max_idle_steps}"
                  f" ({args.evict_max_idle_days}d)"
                  f" combine={args.evict_combine}")
        print(f"         epoch={epoch}  entries={entries_written:,}"
              f"  evicted={entries_evicted:,}   "
              f"before={entries_before:,}")

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
              f"MB/k for Adam 4x buffers)")
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
                flag = f"  +{delta_pct:.0f}% slower"
            elif delta_pct < -5:
                flag = "  (faster)"
            print(f"  {label:6s}: {first3:6.1f}ms -> {last3:6.1f}ms  "
                  f"(d={delta:+.1f}ms, {delta_pct:+.1f}%){flag}")
        print()

    # ── AUC ──
    if snap_auc:
        aucs = [a for _, a in snap_auc if not np.isnan(a)]
        if aucs:
            print(f"AUC:  start={aucs[0]:.4f}  best={max(aucs):.4f}  "
                  f"final={aucs[-1]:.4f}  "
                  f"{'up' if aucs[-1] > aucs[0] else 'down' if aucs[-1] < aucs[0] else '--'}")
        print()

    # ── Weight sanity ──
    sd = model.emb.state_dict()
    w = sd["weight"]
    nan_count  = torch.isnan(w).sum().item()
    inf_count  = torch.isinf(w).sum().item()
    w_min, w_max = w.min().item(), w.max().item()
    w_mean, w_std = w.mean().item(), w.std().item()

    if nan_count > 0 or inf_count > 0:
        print(f"WEIGHT ISSUE: {nan_count} NaN, {inf_count} Inf detected!")
    else:
        print(f"Weights OK:  min={w_min:+.4f}  max={w_max:+.4f}  "
              f"mean={w_mean:+.6f}  std={w_std:+.4f}")

    # ── Verdict ──
    print()
    issues = []
    if nan_count > 0 or inf_count > 0:
        issues.append("NaN/Inf in weights")
    if ent_growth <= 0 and epoch > 1:
        issues.append("No entry growth after first epoch -- "
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
        print("STABILITY PASS -- no issues detected")
    else:
        for issue in issues:
            print(f"WARNING: {issue}")
    print("=" * 70)


if __name__ == "__main__":
    main()
