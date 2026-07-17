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
# =============================================================================
# Parquet reading (pyarrow — lightweight, no TensorFlow dependency)
# =============================================================================
try:
    import pyarrow.parquet as pq
except ImportError:
    print("Error: pyarrow is required for Parquet reading.")
    print("  pip install pyarrow")
    sys.exit(1)

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
BATCH_SIZE      = 1024
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
    N = batch.num_rows
    feat_ids = np.full((N, MAX_FEATS), PAD_VALUE, dtype=np.int64)

    col_offset = 0
    for k in DISCRETE_KEYS:
        feat_ids[:, col_offset] = batch.column(k).to_numpy().astype(np.int64)
        col_offset += 1

    for k in SEQ_KEYS:
        max_len = SEQ_MAX_LENS[k]
        lists = batch.column(k).to_pylist()
        for i, row_list in enumerate(lists):
            if row_list is None or len(row_list) == 0:
                continue
            n_copy = min(len(row_list), max_len)
            feat_ids[i, col_offset:col_offset + n_copy] = row_list[:n_copy]
        col_offset += max_len

    click_col = batch.column("click").to_numpy().astype(np.float32)
    play_score_col = batch.column("play_score").to_numpy().astype(np.float32)
    clk_labels = np.where(play_score_col > 0.1, 1.0, click_col).astype(np.float32)
    play_score_labels = play_score_col / (play_score_col + 1.0)

    return feat_ids, clk_labels, play_score_labels




# =============================================================================
# Streaming Parquet Dataset (IterableDataset for large data)
# =============================================================================
class StreamingParquetDataset(torch.utils.data.IterableDataset):
    """Lazy IterableDataset that streams Parquet files from disk via pyarrow.

    Designed for datasets too large to fit in memory (e.g. 100M+ records).
    Reads Parquet files in configurable row-group-sized chunks using
    ``pq.ParquetFile.iter_batches()``.
    """

    def __init__(self, file_path, max_records=0, seed=42,
                 parquet_batch_size=65536):
        files = []
        print(file_path)
        files = [os.path.join(file_path, f) for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f)) and f.endswith("parquet")]
        if not files:
            raise FileNotFoundError(
                f"No Parquet files found for patterns: {file_path}")
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
        print(my_files)
        records_yielded = 0
        for file_path in my_files:
            pf = pq.ParquetFile(file_path)
            for batch in pf.iter_batches(batch_size=self._parquet_batch_size):
                # print("BATCH \n")
                # print(batch)
                # print("BATCH END...")
                feat_ids, clk_labels, ply_labels = build_feat_ids_from_batch(batch=batch)

                for i in range(len(clk_labels)):
                    yield (torch.from_numpy(feat_ids[i]).long(), 
                        torch.tensor(clk_labels[i], dtype=torch.float32), 
                        torch.tensor(ply_labels[i], dtype=torch.float32)
                        )
                

def collate_fn(batch):
    ids = torch.stack([b[0] for b in batch])
    lbl = torch.tensor([b[1].item() for b in batch], dtype=torch.float32)
    lbp = torch.tensor([b[2].item() for b in batch], dtype=torch.float32)
    return ids, lbl, lbp




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
    def __init__(self, context_dim, tower_units):
        super().__init__()
        self.net = torch.nn.Linear(tower_units, tower_units)
        self.gate = torch.nn.Linear(context_dim, tower_units)
        self.head = torch.nn.Linear(tower_units, 1)

    def forward(self, context, task_out):
        x = torch.relu(self.net(task_out))
        g = torch.sigmoid(self.gate(context))
        return torch.sigmoid(self.head(x * g * 2.0))
    


class PredictModel(torch.nn.Module):
    def __init__(self, emb_dim, token_dim=32, num_heads=4, dff=32):
        super().__init__()
        self.token_dim = token_dim
        self.emb_dim = emb_dim

        self.deep_net = torch.nn.Sequential(
            torch.nn.Linear(emb_dim * N_DISCRETE, 256),
            torch.nn.BatchNorm1d(256)
        )

        self.title_net = torch.nn.Sequential(
            torch.nn.Linear(emb_dim * 5, token_dim),
            torch.nn.BatchNorm1d(token_dim)
        )

        self.tags_net = torch.nn.Sequential(
            torch.nn.Linear(emb_dim * 5, token_dim),
            torch.nn.BatchNorm1d(token_dim)
        )

        self.app_net = torch.nn.Sequential(
            torch.nn.Linear(emb_dim * 30, 128),
            torch.nn.BatchNorm1d(128)
        )

        self.appx_net = torch.nn.Sequential(
            torch.nn.Linear(emb_dim * 30, 128),
            torch.nn.BatchNorm1d(128)
        )

        self.play_seq_net = torch.nn.Sequential(
            torch.nn.Linear(emb_dim * 30, 256),
            torch.nn.BatchNorm1d(256)
        )

        self.cp_seq_net = torch.nn.Sequential(
            torch.nn.Linear(emb_dim * 10, 256),
            torch.nn.BatchNorm1d(256)
        )

        self.real_clk_net = torch.nn.Sequential(
            torch.nn.Linear(emb_dim * 10, 256),
            torch.nn.BatchNorm1d(256)
        )

        self.real_play_net = torch.nn.Sequential(
            torch.nn.Linear(emb_dim * 10, 256),
            torch.nn.BatchNorm1d(256)
        )

        self.transformer = torch.nn.TransformerEncoderLayer(
            d_model=token_dim, nhead=num_heads, dim_feedforward=dff,
            dropout=0.1, activation='gelu', batch_first=True)
        self.att_pool = AttentionPooling(token_dim)
        n_context = token_dim
        self.mmoe = MMoE(num_tasks=2, num_experts=16, expert_units=128, input_dim=n_context)
        self.ppnet_click = PPNet(context_dim= token_dim,tower_units=128)
        self.ppnet_play = PPNet(context_dim= token_dim, tower_units=128)


    def forward(self, feat_embs):
        B = feat_embs.size(0)

        deep_embs = feat_embs[:, :N_DISCRETE, :].reshape(B, -1)
        deep_out = self.deep_net(deep_embs)
        n_deep_tokens = 256 // self.token_dim
        token_deep = deep_out.reshape(B, n_deep_tokens, self.token_dim)

        col = N_DISCRETE
        seq_emb_dict = {}
        for key, slen in SEQ_MAX_LENS.items():
            block = feat_embs[:, col:col+slen, :].reshape(B, -1)
            seq_emb_dict[key] = torch.nan_to_num(block, nan=0.0)

        title_out = self.title_net(seq_emb_dict["dis_63"])
        n_title_tokens = 32 // self.token_dim
        token_title = title_out.reshape(B, n_title_tokens, self.token_dim)

        tags_out = self.tags_net(seq_emb_dict["dis_64"])
        n_tags_tokens = 32 // self.token_dim
        token_tags = tags_out.reshape(B, n_tags_tokens, self.token_dim)

        app_out = self.app_net(seq_emb_dict["dis_65"])
        n_app_tokens = 128 // self.token_dim
        token_app = app_out.reshape(B, n_app_tokens, self.token_dim)

        appx_out = self.appx_net(seq_emb_dict["dis_66"])
        n_appx_tokens = 128 // self.token_dim
        token_appx = appx_out.reshape(B, n_appx_tokens, self.token_dim)

        play_seq_out = self.play_seq_net(seq_emb_dict["dis_67"])
        n_play_tokens = 256 // self.token_dim
        token_play_seq = play_seq_out.reshape(B, n_play_tokens, self.token_dim)

        cp_seq_out = self.cp_seq_net(seq_emb_dict["dis_68"])
        n_cp_tokens = 256 // self.token_dim
        token_cp_seq = cp_seq_out.reshape(B, n_cp_tokens, self.token_dim)

        real_clk_out = self.real_clk_net(seq_emb_dict["dis_69"])
        n_real_clk_tokens = 256 // self.token_dim
        token_real_clk = real_clk_out.reshape(B, n_real_clk_tokens, self.token_dim)

        real_play_out = self. real_play_net(seq_emb_dict["dis_70"])
        n_real_play_tokens = 256 // self.token_dim
        token_real_play = real_play_out.reshape(B, n_real_play_tokens, self.token_dim)


        tokens_list = [token_deep, token_title, token_tags, token_app, token_appx, token_play_seq, token_cp_seq, token_real_clk, token_real_play]
        tokens_cat = torch.cat(tokens_list, dim=1)

        all_tokens = self.transformer(tokens_cat)
        all_pooled = self.att_pool(all_tokens)

        task_outputs = self.mmoe(all_pooled)

        click = self.ppnet_click(all_pooled, task_outputs[0])

        play = self.ppnet_play(all_pooled, task_outputs[1])

        return click.squeeze(-1), play.squeeze(-1)

class EmbeddingModel(torch.nn.Module):
    def __init__(self, emb_dim, capacity, lr, block_size):
        super().__init__()
        self.emb = HashEmbedding(
            emb_dim, capacity,
            optimizer="adam", lr=lr,
            initial_scale=0.01,
            block_size=block_size,
        )


    def forward(self, feat_ids):
        embs = self.emb(feat_ids) 
        mask = (feat_ids == -1).unsqueeze(-1)   # (B, F, 1)
        embs = embs.masked_fill(mask, float('nan'))
        return embs
    
    def step(self):
        self.emb.step()


class UnifiedModel(torch.nn.Module):
    def __init__(self, emb_dim, capacity, lr, block_size, token_dim=32, num_heads=4, dff=32):
        super().__init__()
        self.emb = HashEmbedding(
            emb_dim, capacity,
            optimizer="adam", lr=lr,
            initial_scale=0.01,
            block_size=block_size,
        )
        # self.emb_model = EmbeddingModel(emb_dim=emb_dim, capacity=capacity, lr=lr, block_size=block_size)
        self.pred_model = PredictModel(emb_dim=emb_dim, token_dim=token_dim)

    def forward(self, feat_ids):
        feat_embs = self.emb(feat_ids)
        # feat_embs.register_hook(lambda g: print(f"  emb grad norm: {g.norm().item():.6f}"))
        
        mask = (feat_ids == -1).unsqueeze(-1)   # (B, F, 1)
        feat_embs = feat_embs.masked_fill(mask, float('nan'))

        feat_embs = feat_embs.to(next(self.pred_model.parameters()).device)
        clk, play = self.pred_model(feat_embs)

        return clk, play

    def step(self):
        self.emb.step()

# =============================================================================
# Helpers
# =============================================================================
def mem_rss_mb():
    """Current RSS in MB (Linux: ru_maxrss reports KB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0



def main():
    parser = argparse.ArgumentParser(
        description="HashEmb stability test with real Parquet data")
    parser.add_argument("--data", type=str, required=True,
                        help="Parquet file pattern(s), e.g. 'data/*.parquet'")
    parser.add_argument("--val-data", type=str, default=None,
                        help="Separate validation Parquet file pattern(s) "
                             "(default: no validation)")
    
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    
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
    train_ds = StreamingParquetDataset(
        args.data,
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
    print(f"[MEM] After dataset init:  {mem1:.0f} MB  "
          f"(+{mem1 - mem0:.0f} MB)")
    print()


    # data_path = "/Users/fanwei/study/HKV/bill_data/"
    # train_ds = StreamingParquetDataset(
    #     data_path,
    #     max_records=MAX_RECORDS,
    #     seed=SEED,
        
    # )

    # train_loader = DataLoader(
    #     train_ds, batch_size=BATCH_SIZE,
    #     collate_fn=collate_fn, drop_last=True,
        
    # )

    mem2 = mem_rss_mb()
    print(f"[MEM] Before model: {mem2:.0f} MB")

    model = UnifiedModel(emb_dim=EMBEDDING_DIM, capacity=capacity, lr=lr, block_size=block_size)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model.pred_model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.008)
    resume_epoch = 0
    if args.resume:
        binary_path = args.resume.replace('.pt', '.hashemb')
        if os.path.exists(binary_path):
            t_load = time.time()
            model.emb.load(binary_path)
            dt_load = time.time() - t_load
            ckpt = torch.load(args.resume, map_location=device, weights_only=True)
            model.pred_model.load_state_dict(ckpt["dense"])
            optimizer.load_state_dict(ckpt["opt"])
            resume_epoch = ckpt.get("epoch", 0)
            print(f"  [RESUME] Hash table from {os.path.basename(binary_path)}"
                  f"  ({dt_load:.1f}s)", flush=True)
            print(f"           Dense model from {os.path.basename(args.resume)}")
            print(f"           prev_epoch={resume_epoch}  "
                  f"entries={model.emb.num_entries:,}")
        else:
            print(f"  [RESUME] {binary_path} not found, cold start")

    mem3 = mem_rss_mb()
    print(f"  Initial entries: {model.emb.num_entries:,}")
    print(f"[MEM] After model:  {mem3:.0f} MB  (+{mem3 - mem2:.0f} MB)")
    print()
    
    loss_clk = torch.nn.BCELoss()
    loss_play = torch.nn.BCELoss()
    total_loss = 0
    # 累积器：每个 batch 的预估 & 标签
    clk_preds, clk_labels = [], []
    play_preds, play_labels = [], []

    wall_start  = time.time()
    t_prev_end = time.perf_counter()
    batch_this_epoch = 0
    for bi, (feat_ids, lbk, lbp) in enumerate(train_loader):
        t_fetch = time.perf_counter()
        dt_data = t_fetch - t_prev_end
        # print(f"bi: {bi}")
        # print(chuck)
        # print(clk_labels)
        # print(ply_labels)
        batch_this_epoch += 1
        t0 = time.perf_counter()
        clk, play = model(feat_ids)
        lbk = lbk.to(device)
        lbp = lbp.to(device)

        t1 = time.perf_counter()

        dt_fwd = t1 - t0
        # print(clk)
        # print(play)
        loss = 0.5*loss_clk(clk, lbk) + 0.5*loss_play(play, lbp)
        # if bi > 1:
        #     break
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()           # dense params 更新
        model.step()     # hash table 更新（内部自动清零 grad）

        t2 = time.perf_counter()

        dt_bwd = t2 - t1 

        dt_per_batch = t2 - t0 

        
        total_loss += loss.item()
        # 累积 numpy（detach + cpu）
        clk_preds.append(clk.detach().cpu().numpy())
        clk_labels.append(lbk.detach().cpu().numpy())
        play_preds.append(play.detach().cpu().numpy())
        play_labels.append(lbp.detach().cpu().numpy())

        t_prev_end = time.perf_counter()
        if bi % 50 == 0 and bi > 0:
            # print(f"batch {bi}: loss={loss.item():.4f}")
            cp = np.concatenate(clk_preds)
            cl = np.concatenate(clk_labels)
            pp = np.concatenate(play_preds)
            pl = np.concatenate(play_labels)

            # ── 诊断打印 ──
            # print(f"--- bi={bi} (window size={len(cl)}) ---")
            # print(f"  click_label: pos={cl.sum():.0f} neg={len(cl)-cl.sum():.0f} "
            #         f"| pred mean={cp.mean():.4f} std={cp.std():.4f} min={cp.min():.4f} max={cp.max():.4f}")
            # print(f"  play_label:  mean={pl.mean():.4f} std={pl.std():.4f} min={pl.min():.4f} max={pl.max():.4f} "
            #         f"| >0.5={int((pl>0.5).sum())} <=0.5={int((pl<=0.5).sum())}")
            # print(f"  play_pred:   mean={pp.mean():.4f} std={pp.std():.4f} min={pp.min():.4f} max={pp.max():.4f}")


            auc_click = roc_auc_score(cl.astype(int), cp)

            # play label 二值化：以中位数为阈值
            thresh = 0.5
            pl_bin = (pl > thresh).astype(int)
            auc_play = roc_auc_score(pl_bin, pp)

            print(f"bi={bi:4d} | data={dt_data*1000.0:.1f}ms | fwd={dt_fwd*1000.0:.1f}ms | bwd={dt_bwd*1000.0:.1f}ms | perBatch={dt_per_batch*1000.0:.1f}ms | loss={loss.item():.4f} | click_auc={auc_click:.4f} | play_auc={auc_play:.4f}")

            # 清空累积器
            clk_preds, clk_labels = [], []
            play_preds, play_labels = [], []

    # ── Validation ──
    if val_loader is not None:
        model.eval()
        # all_scores, all_labels_gt = [], []
        rlbk_list, rlbp_list = [], []
        plbk_list, plbp_list = [], []
        with torch.no_grad():
            for feat_ids, lbk, lbp in val_loader:
                plbk, plbp = model(feat_ids)
                rlbk_list.append(lbk.detach().cpu().numpy())
                plbk_list.append(plbk.detach().cpu().numpy())

                rlbp_list.append(lbp.detach().cpu().numpy())
                plbp_list.append(plbp.detach().cpu().numpy())
                # all_scores.append(torch.sigmoid(logits).numpy())
                # all_labels_gt.append(labels.numpy())

        try:
            clk_auc = roc_auc_score(
                np.concatenate(rlbk_list),
                np.concatenate(plbk_list))
            
            # play label 二值化：以中位数为阈值
            thresh = 0.5
            rlbp_np = np.concatenate(rlbp_list)
            pl_bin = (rlbp_np > thresh).astype(int)
            play_auc = roc_auc_score(pl_bin, np.concatenate(plbp_list))

            print(f"EVAL AUC: clk_auc:{clk_auc:.6f}, play_auc:{play_auc:.6f}")
        except ValueError:
            auc = 0.5
    else:
        auc = float("nan")
    
    total_time = time.time() - wall_start
    print(f" Total Train Time: {total_time:.1f}s")

    # ── Save checkpoint ──
    # HashEmb → C++ binary (bucket-by-bucket, zero extra memory).
    # Dense + optimizer → torch.save (tens of MB).
    if args.save:
        binary_path = args.save.replace('.pt', '.hashemb')
        max_idle_steps = args.evict_max_idle_days * (batch_this_epoch or 0) if args.evict_max_idle_days > 0 else 0
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
            "dense": model.pred_model.state_dict(),
            "opt": optimizer.state_dict(),
            "epoch": resume_epoch + 1,
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
        print(f"         epoch={resume_epoch+1}  entries={entries_written:,}"
              f"  evicted={entries_evicted:,}   "
              f"before={entries_before:,}")


if __name__ == "__main__":
    main()