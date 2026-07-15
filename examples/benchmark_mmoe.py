#!/usr/bin/env python3

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

try:
    import pyarrow.parquet as pq
except ImportError:
    print("Error: pyarrow is required for Parquet reading.")
    print("  pip install pyarrow")
    sys.exit(1)

# ── Constants ──
N_DISCRETE = 63
SEQ_MAX_LENS = {
    "dis_63": 5, "dis_64": 5, "dis_65": 30, "dis_66": 30,
    "dis_67": 30, "dis_68": 10, "dis_69": 10, "dis_70": 10,
    "dis_71": 10, "dis_72": 10, "dis_73": 10, "dis_74": 10,
}
SEQ1_COLS = [("dis_63", 5), ("dis_64", 5), ("dis_71", 10),
             ("dis_72", 10), ("dis_74", 10)]
PAD_VALUE = -1
MAX_FEATS = N_DISCRETE + sum(v for v in SEQ_MAX_LENS.values())
EMBEDDING_DIM = 16
HASH_CAPACITY = 20_000_000
BLOCK_SIZE = 1_000_000
BATCH_SIZE = 4096
EPOCHS = 1
MAX_RECORDS = 0
LR = 0.01

# ── Parquet → feat_ids + labels ──
def _make_feat_ids(batch):
    max_len = N_DISCRETE + sum(SEQ_MAX_LENS.values())
    feat_ids = np.full((batch.num_rows, max_len), PAD_VALUE, dtype=np.int64)
    col_offset = 0
    for i in range(N_DISCRETE):
        col_name = f"dis_{i:02d}"
        col = batch.column(col_name).to_numpy().astype(np.int64)
        feat_ids[:, col_offset] = col
        col_offset += 1
    for col_name, max_len_seq in SEQ_MAX_LENS.items():
        lists = batch.column(col_name).to_pylist()
        for i, row_list in enumerate(lists):
            if row_list is None or len(row_list) == 0:
                continue
            n_copy = min(len(row_list), max_len_seq)
            feat_ids[i, col_offset:col_offset + n_copy] = row_list[:n_copy]
        col_offset += max_len_seq
    click_col = batch.column("click").to_numpy().astype(np.float32)
    play_score_col = batch.column("play_score").to_numpy().astype(np.float32)
    labels = np.where(play_score_col > 0.1, 1.0, click_col).astype(np.float32)
    return feat_ids, labels

# ── Streaming dataset ──
class StreamingParquetDataset(torch.utils.data.IterableDataset):
    def __init__(self, file_patterns, max_records=0, seed=42, parquet_batch_size=65536):
        files = []
        for pattern in file_patterns:
            matched = glob.glob(pattern)
            files.extend(matched)
        if not files:
            raise FileNotFoundError(f"No Parquet files found for patterns: {file_patterns}")
        files.sort()
        self._files = files
        self._max_records = max_records
        self._seed = seed
        self._parquet_batch_size = parquet_batch_size
        self._epoch = 0
        self.n_files = len(files)

    def set_epoch(self, epoch):
        self._epoch = epoch

    def __iter__(self):
        rng = np.random.RandomState(self._seed + self._epoch)
        files = self._files.copy()
        rng.shuffle(files)
        records_yielded = 0
        for fp in files:
            pf = pq.ParquetFile(fp)
            for batch in pf.iter_batches(batch_size=self._parquet_batch_size):
                feat_ids, labels = _make_feat_ids(batch)
                n = feat_ids.shape[0]
                for j in range(n):
                    if self._max_records > 0 and records_yielded >= self._max_records:
                        return
                    records_yielded += 1
                    yield feat_ids[j], labels[j]

def collate_fn(batch):
    ids = torch.stack([torch.as_tensor(b[0]) for b in batch])
    lbl = torch.tensor([b[1].item() for b in batch], dtype=torch.float32)
    return ids, lbl

# ── Model ──
class MMoE(torch.nn.Module):
    def __init__(self, num_tasks, num_experts, expert_units, input_dim):
        super().__init__()
        self.experts = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(input_dim, expert_units * 4),
                torch.nn.GELU(),
                torch.nn.Linear(expert_units * 4, expert_units),
                torch.nn.GELU(),
            ) for _ in range(num_experts)
        ])
        self.gates = torch.nn.ModuleList([
            torch.nn.Linear(input_dim, num_experts) for _ in range(num_tasks)
        ])

    def forward(self, x):
        expert_outputs = torch.stack([e(x) for e in self.experts], dim=1)
        task_outputs = []
        for i in range(len(self.gates)):
            gate = torch.softmax(self.gates[i](x), dim=-1).unsqueeze(-1)
            task_out = (expert_outputs * gate).sum(dim=1)
            task_outputs.append(task_out)
        return task_outputs

class AttentionPooling(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.att = torch.nn.Linear(dim, 1)

    def forward(self, x, mask=None):
        scores = self.att(x)
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(-1), -1e9)
        weights = torch.softmax(scores, dim=1)
        return (x * weights).sum(dim=1)

class PPNet(torch.nn.Module):
    def __init__(self, tower_units):
        super().__init__()
        self.net = torch.nn.Linear(tower_units, tower_units)
        self.gate = torch.nn.Linear(tower_units, tower_units)
        self.head = torch.nn.Linear(tower_units, 1)

    def forward(self, context, task_out):
        x = torch.relu(self.net(task_out))
        g = torch.sigmoid(self.gate(context))
        return torch.sigmoid(self.head(x * g * 2.0))

class MMoEModel(torch.nn.Module):
    def __init__(self, emb_dim, capacity, lr, block_size, token_dim=32, num_heads=4, dff=32):
        super().__init__()
        self.emb = HashEmbedding(
            emb_dim, capacity, optimizer="adam", lr=lr,
            initial_scale=0.01, block_size=block_size)
        self.token_dim = token_dim
        self.emb_dim = emb_dim

        self.deep_net = torch.nn.Sequential(
            torch.nn.Linear(emb_dim * N_DISCRETE, 256),
            torch.nn.BatchNorm1d(256))

        n_seq1 = len(SEQ1_COLS)
        self.seq_mean_net = torch.nn.Sequential(
            torch.nn.Linear(emb_dim * n_seq1, 64),
            torch.nn.BatchNorm1d(64))

        g1_len = sum(SEQ_MAX_LENS[c] for c in ["dis_65", "dis_66", "dis_67"])
        g2_len = sum(SEQ_MAX_LENS[c] for c in ["dis_68", "dis_69", "dis_70", "dis_73"])
        self.g1_len = g1_len
        self.g2_len = g2_len
        self.token_proj1 = torch.nn.Linear(emb_dim, token_dim)
        self.token_proj2 = torch.nn.Linear(emb_dim, token_dim)
        self.down1 = torch.nn.AvgPool1d(kernel_size=4, stride=2)
        self.down2 = torch.nn.AvgPool1d(kernel_size=4, stride=2)

        self.transformer = torch.nn.TransformerEncoderLayer(
            d_model=token_dim, nhead=num_heads, dim_feedforward=dff,
            dropout=0.1, activation='gelu', batch_first=True)
        self.att_pool = AttentionPooling(token_dim)

        n_context = token_dim + emb_dim * 5  # att_pool + 5 context feats
        self.mmoe = MMoE(num_tasks=2, num_experts=16, expert_units=32, input_dim=n_context)
        self.ppnet_click = PPNet(tower_units=128)
        self.ppnet_play = PPNet(tower_units=128)

    def forward(self, feat_ids):
        B = feat_ids.size(0)
        mask = (feat_ids != PAD_VALUE).float()
        embs = self.emb(feat_ids) * mask.unsqueeze(-1)

        # Deep tower
        deep_embs = embs[:, :N_DISCRETE, :].reshape(B, -1)
        deep_out = self.deep_net(deep_embs)
        n_deep_tokens = 256 // self.token_dim
        token_deep = deep_out.reshape(B, n_deep_tokens, self.token_dim)

        # Seq-mean tower
        col = N_DISCRETE
        seq_mean_parts = []
        for _, slen in SEQ1_COLS:
            block = embs[:, col:col+slen, :]
            mask_block = mask[:, col:col+slen].unsqueeze(-1)
            val = (block * mask_block).sum(dim=1) / mask_block.sum(dim=1).clamp(min=1)
            seq_mean_parts.append(val)
            col += slen
        seq_mean_cat = torch.cat(seq_mean_parts, dim=-1)
        seq_mean_out = self.seq_mean_net(seq_mean_cat)
        n_seq_tokens = 64 // self.token_dim
        token_seq = seq_mean_out.reshape(B, n_seq_tokens, self.token_dim)

        # Seq-token group1 & group2
        col1 = col
        g1_embs = embs[:, col1:col1+self.g1_len, :]
        g1_tokens = self.token_proj1(g1_embs)
        g1_tokens = self.down1(g1_tokens.permute(0, 2, 1)).permute(0, 2, 1)

        col2 = col1 + self.g1_len
        g2_embs = embs[:, col2:col2+self.g2_len, :]
        g2_tokens = self.token_proj2(g2_embs)
        g2_tokens = self.down2(g2_tokens.permute(0, 2, 1)).permute(0, 2, 1)

        # Transformer + Pool
        all_tokens = torch.cat([token_deep, token_seq, g1_tokens, g2_tokens], dim=1)
        all_tokens = self.transformer(all_tokens)
        att_pooled = self.att_pool(all_tokens)

        # Context: dis_04(scene), dis_07(ram), dis_41(expo), dis_42(clk), dis_43(visit)
        ctx_idxs = [4, 7, 41, 42, 43]
        ctx_embs = embs[:, ctx_idxs, :].reshape(B, -1)

        # MMoE + PPNet
        task_input = torch.cat([att_pooled, ctx_embs], dim=-1)
        task_outputs = self.mmoe(task_input)
        click = self.ppnet_click(task_input, task_outputs[0])
        play = self.ppnet_play(task_input, task_outputs[1])
        return click.squeeze(-1), play.squeeze(-1)

    def step(self):
        self.emb.step()

# ── Helpers ──
def mem_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

# ── Main ──
def main():
    parser = argparse.ArgumentParser(description="HashEmb MMoE benchmark")
    parser.add_argument("--data", nargs="+", required=True, help="Parquet file pattern(s)")
    parser.add_argument("--val-data", nargs="+", default=None)
    parser.add_argument("--max-records", type=int, default=MAX_RECORDS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--steps", type=int, default=EPOCHS)
    parser.add_argument("--duration", type=int, default=0)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--capacity", type=int, default=HASH_CAPACITY)
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--stats-samples", type=int, default=10000)
    parser.add_argument("--parquet-batch-size", type=int, default=65536)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--save", type=str, default="")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--evict-min-count", type=int, default=0)
    parser.add_argument("--evict-max-idle-days", type=int, default=0)
    parser.add_argument("--evict-combine", type=str, default="and")
    args = parser.parse_args()

    batch_size = args.batch_size
    max_records = args.max_records
    epochs = args.steps if args.duration <= 0 else 999999
    duration = args.duration
    emb_dim = EMBEDDING_DIM
    capacity = args.capacity
    block_size = args.block_size

    print("=" * 70)
    print("HashEmb MMoE Benchmark — Real Parquet Data")
    print("=" * 70)
    print(f"  Data pattern:          {args.data}")
    print(f"  Max records/epoch:     {max_records:,}")
    print(f"  Batch size:            {batch_size:,}")
    print(f"  Parquet chunk size:    {args.parquet_batch_size:,}")
    print(f"  DataLoader workers:    {args.num_workers}")
    print(f"  Feat IDs / sample:     <= {MAX_FEATS}")
    print(f"  Embedding dim:         {emb_dim}")
    print(f"  Hash capacity:         {capacity:,}")
    print(f"  Block size:            {block_size:,}")
    print(f"  Optimizer:             Adam, lr={args.lr}")
    print(f"  Epochs:                {epochs}")
    if args.save:
        print(f"  Save to:               {args.save}")
    if args.debug:
        print(f"  Debug:                 True")
    print()

    mem0 = mem_rss_mb()
    print(f"[MEM] Before load: {mem0:.0f} MB")

    # ── Stats estimation ──
    try:
        ds_sample = StreamingParquetDataset(args.data, max_records=args.stats_samples,
                                            parquet_batch_size=args.parquet_batch_size)
        feat_ids_sample = []
        for feat_id, _ in ds_sample:
            feat_ids_sample.append(feat_id)
            if len(feat_ids_sample) >= args.stats_samples:
                break
        feat_ids_sample = np.array(feat_ids_sample)
        valid_feats = (feat_ids_sample != PAD_VALUE).sum(axis=1).mean()
        feat_min, feat_max = feat_ids_sample.min(), feat_ids_sample.max()
        print(f"  Stats estimated from {len(feat_ids_sample):,} records")
        print(f"    Est. avg valid feats/sample: {valid_feats:.1f}")
        print(f"    Est. feat_id range: [{feat_min:,}, {feat_max:,}]")
    except Exception as e:
        print(f"  Stats estimation skipped: {e}")

    # ── Dataset ──
    train_ds = StreamingParquetDataset(args.data, max_records=max_records,
                                       parquet_batch_size=args.parquet_batch_size)
    loader = DataLoader(train_ds, batch_size=batch_size, collate_fn=collate_fn,
                        num_workers=args.num_workers, prefetch_factor=args.prefetch_factor)
    mem1 = mem_rss_mb()
    print(f"[MEM] After dataset init:  {mem1:.0f} MB  (+{mem1 - mem0:.0f} MB)")

    # ── Validation ──
    val_loader = None
    if args.val_data:
        val_ds = StreamingParquetDataset(args.val_data, max_records=0,
                                         parquet_batch_size=args.parquet_batch_size)
        val_loader = DataLoader(val_ds, batch_size=batch_size, collate_fn=collate_fn,
                                num_workers=args.num_workers, prefetch_factor=args.prefetch_factor)

    # ── Model ──
    print(f"\n[MEM] Before model: {mem1:.0f} MB")
    model = MMoEModel(emb_dim, capacity, lr=args.lr, block_size=block_size)
    opt = torch.optim.Adam([p for n, p in model.named_parameters() if "emb" not in n], lr=args.lr)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # ── Resume ──
    resume_epoch = 0
    if args.resume:
        binary_path = args.resume.replace('.pt', '.hashemb')
        if os.path.exists(binary_path):
            t_load = time.time()
            model.emb.load(binary_path)
            dt_load = time.time() - t_load
            ckpt = torch.load(args.resume, map_location="cpu", weights_only=True)
            model.load_state_dict(ckpt["dense"], strict=False)
            opt.load_state_dict(ckpt["opt"])
            resume_epoch = ckpt.get("epoch", 0)
            print(f"  [RESUME] Hash table from {os.path.basename(binary_path)}  ({dt_load:.1f}s)", flush=True)
            print(f"           Dense model from {os.path.basename(args.resume)}")
            print(f"           prev_epoch={resume_epoch}  entries={model.emb.num_entries:,}")
        else:
            print(f"  [RESUME] {binary_path} not found, cold start")

    mem2 = mem_rss_mb()
    print(f"  Initial entries: {model.emb.num_entries:,}")
    print(f"[MEM] After model:  {mem2:.0f} MB  (+{mem2 - mem1:.0f} MB)")
    print()

    # ── Training ──
    n_batches = None
    snap_mem, snap_auc = [], []
    prev_mem = mem2
    prev_ent = 0
    wall_start = time.time()
    lookups_per_batch = batch_size * MAX_FEATS

    print("-" * 95)
    print(f"{'Ep':>4s} | {'loss':>7s} {'auc':>7s} | {'entries':>10s} {'RSS(MB)':>8s} {'dMB':>6s} | "
          f"{'fwd':>6s} {'bwd':>6s} {'step':>6s} {'total':>6s} | {'Mlookup/s':>9s}")
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
        ep_loss, ep_fwd, ep_bwd, ep_step, ep_tot = 0.0, [], [], [], []
        n_batches_this_epoch = 0
        int_loss, int_fwd, int_bwd, int_step = 0.0, [], [], []

        for step_idx, (feat_ids, labels) in enumerate(loader):
            t_data = time.perf_counter()
            feat_ids = feat_ids.to(device)
            labels = labels.to(device)

            t0 = time.perf_counter()
            click, play = model(feat_ids)
            loss = F.binary_cross_entropy_with_logits(click, labels)
            t1 = time.perf_counter()

            opt.zero_grad()
            loss.backward()
            t2 = time.perf_counter()

            model.step()
            opt.step()
            t3 = time.perf_counter()

            dt_data = t0 - t_data
            dt_fwd, dt_bwd, dt_step = t1 - t0, t2 - t1, t3 - t2
            dt_comp = dt_fwd + dt_bwd + dt_step

            ep_loss += loss.item()
            ep_fwd.append(dt_fwd)
            ep_bwd.append(dt_bwd)
            ep_step.append(dt_step)
            ep_tot.append(dt_comp)
            n_batches_this_epoch += 1

            if (step_idx + 1) % args.log_interval == 0:
                int_loss += loss.item()
                int_fwd.append(dt_fwd)
                int_bwd.append(dt_bwd)
                int_step.append(dt_step)
                n_ent = model.emb.num_entries
                tput = lookups_per_batch / max(dt_comp, 1e-9) / 1e6
                avg_dt_data = dt_data * 1000
                avg_comp = dt_comp * 1000
                avg_fwd = dt_fwd * 1000
                avg_bwd = dt_bwd * 1000
                avg_step = dt_step * 1000
                print(f"  [Ep {epoch}, bat {step_idx + 1:<7d}] "
                      f"loss={loss.item():.4f} auc=--  ent={n_ent:,}  "
                      f"data={avg_dt_data:.0f}ms comp={avg_comp:.0f}ms"
                      f"(fwd={avg_fwd:.0f} bwd={avg_bwd:.0f} step={avg_step:.0f})  "
                      f"bat={avg_dt_data + avg_comp:.0f}ms  tput={tput:.1f}M/s")

        # Epoch summary
        n_ent = model.emb.num_entries
        cur_mem = mem_rss_mb()
        mem_delta = cur_mem - prev_mem
        ent_delta = n_ent - prev_ent
        avg_fwd = np.mean(ep_fwd) * 1000
        avg_bwd = np.mean(ep_bwd) * 1000
        avg_step = np.mean(ep_step) * 1000
        avg_tot = np.mean(ep_tot) * 1000
        tput = lookups_per_batch / (avg_tot / 1000) / 1e6
        if n_batches is None:
            n_batches = n_batches_this_epoch
            print(f"  Discovered {n_batches} batches/epoch from first epoch")
        avg_loss = ep_loss / max(n_batches_this_epoch, 1)
        print(f"{epoch:4d} | {avg_loss:7.4f}      -- | {n_ent:10,d} {cur_mem:8.0f} {mem_delta:+6.0f} | "
              f"{avg_fwd:6.1f} {avg_bwd:6.1f} {avg_step:6.1f} {avg_tot:6.1f} | {tput:9.1f}")
        snap_mem.append((epoch, cur_mem, n_ent))
        prev_mem = cur_mem
        prev_ent = n_ent

    total_time = time.time() - wall_start

    # ── Save ──
    if args.save:
        binary_path = args.save.replace('.pt', '.hashemb')
        max_idle_steps = args.evict_max_idle_days * (n_batches or 0) if args.evict_max_idle_days > 0 else 0
        print(f"  [SAVE] Writing hash table to {binary_path} ...", flush=True)
        entries_before = model.emb.num_entries
        t_save = time.time()
        entries_written = model.emb.save(binary_path,
                          min_count=args.evict_min_count,
                          max_idle_steps=max_idle_steps,
                          combine=args.evict_combine)
        dt_save = time.time() - t_save
        entries_evicted = entries_before - entries_written
        print(f"  [SAVE] Hash table done ({entries_written:,} entries, {entries_evicted:,} evicted)  "
              f"[{dt_save:.1f}s]", flush=True)
        print(f"  [SAVE] Writing dense model to {args.save} ...", flush=True)
        dense_ckpt = {
            "dense": {k: v for k, v in model.state_dict().items() if "emb" not in k},
            "opt": opt.state_dict(),
            "epoch": epoch,
        }
        torch.save(dense_ckpt, args.save)
        print(f"  [SAVE] Dense model done", flush=True)

    # ── Summary ──
    print()
    print("=" * 70)
    print("Stability Summary")
    print("=" * 70)
    print(f"  Total epochs:   {epoch}")
    print(f"  Total time:     {total_time:.1f}s ({total_time / max(epoch, 1):.1f}s/epoch)")
    print(f"  Total batches:  {epoch * (n_batches or 0):,}")
    print(f"  Total lookups:  {epoch * (n_batches or 0) * batch_size * MAX_FEATS:,}")
    print()

if __name__ == "__main__":
    main()
