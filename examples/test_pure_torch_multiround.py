#!/usr/bin/env python3
"""
Pure PyTorch multi-round save/load resume test.

Train nn.Embedding baseline on ML-1M with multiple save/load rounds
to see if AUC keeps rising (overfitting) across checkpoints.

Usage:
    python examples/test_pure_torch_multiround.py [--rounds N] [--epochs_per_round N]
"""
import os
import sys
import time
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ===========================================================================
# Config
# ===========================================================================
EMBEDDING_DIM = 16
BATCH_SIZE = 1024
LR = 0.01
SEED = 42

DATA_PATH = "/Users/rexus/works/hkv_pproject/ml-1m/ratings.dat"
N_USERS = 6040
N_MOVIES = 3952

CKPT_DIR = "/tmp/pure_torch_multiround"

torch.manual_seed(SEED)
np.random.seed(SEED)


# ===========================================================================
# Data
# ===========================================================================
class ML1M_Dataset(Dataset):
    def __init__(self, path, train=True, split=0.8, seed=SEED):
        data = []
        with open(path) as f:
            for line in f:
                uid, mid, rating, _ = line.strip().split("::")
                uid = int(uid); mid = int(mid)
                label = 1 if int(rating) >= 4 else 0
                data.append((uid, mid, label))
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(data))
        n_train = int(len(data) * split)
        indices = perm[:n_train] if train else perm[n_train:]
        self.users = np.array([data[i][0] for i in indices], dtype=np.int64)
        self.movies = np.array([data[i][1] for i in indices], dtype=np.int64)
        self.labels = np.array([data[i][2] for i in indices], dtype=np.float32)

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        return {"user_id": self.users[idx], "movie_id": self.movies[idx],
                "label": self.labels[idx]}


def collate_fn(batch):
    users = torch.tensor([b["user_id"] for b in batch], dtype=torch.int64)
    movies = torch.tensor([b["movie_id"] for b in batch], dtype=torch.int64)
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.float32)
    return {"user_id": users, "movie_id": movies, "label": labels}


# ===========================================================================
# Pure PyTorch Model
# ===========================================================================
class PureTorchModel(torch.nn.Module):
    def __init__(self, n_users, n_movies, embedding_dim):
        super().__init__()
        self.user_emb = torch.nn.Embedding(n_users + 1, embedding_dim, padding_idx=0)
        self.movie_emb = torch.nn.Embedding(n_movies + 1, embedding_dim, padding_idx=0)
        total_dim = 2 * embedding_dim
        self.predict = torch.nn.Linear(total_dim, 1)

        # Init weights
        torch.nn.init.normal_(self.user_emb.weight, mean=0, std=0.01)
        torch.nn.init.normal_(self.movie_emb.weight, mean=0, std=0.01)
        torch.nn.init.xavier_uniform_(self.predict.weight)
        torch.nn.init.zeros_(self.predict.bias)

    def forward(self, feat_dict):
        u_emb = self.user_emb(feat_dict["user_id"])
        m_emb = self.movie_emb(feat_dict["movie_id"])
        combined = torch.cat([u_emb, m_emb], dim=-1)
        return self.predict(combined).squeeze(-1)


# ===========================================================================
# Train / Evaluate
# ===========================================================================
def train_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        feat_dict = {"user_id": batch["user_id"], "movie_id": batch["movie_id"]}
        labels = batch["label"]
        optimizer.zero_grad()
        logits = model(feat_dict)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_scores, all_labels = [], []
    for batch in loader:
        feat_dict = {"user_id": batch["user_id"], "movie_id": batch["movie_id"]}
        logits = model(feat_dict)
        scores = torch.sigmoid(logits)
        all_scores.append(scores.cpu().numpy())
        all_labels.append(batch["label"].cpu().numpy())
    return roc_auc_score(np.concatenate(all_labels), np.concatenate(all_scores))


# ===========================================================================
# Save / Load helpers
# ===========================================================================
CKPT_MODEL = "model.pt"
CKPT_OPT = "optim.pt"
CKPT_META = "meta.txt"


def save_checkpoint(round_idx, epoch, model, optimizer, loss, auc):
    os.makedirs(f"{CKPT_DIR}/round_{round_idx:03d}", exist_ok=True)
    path = f"{CKPT_DIR}/round_{round_idx:03d}"
    torch.save(model.state_dict(), f"{path}/{CKPT_MODEL}")
    torch.save(optimizer.state_dict(), f"{path}/{CKPT_OPT}")
    with open(f"{path}/{CKPT_META}", "w") as f:
        f.write(f"round={round_idx}\nepoch={epoch}\nloss={loss:.6f}\nauc={auc:.6f}\n")


def load_checkpoint(round_idx, model, optimizer):
    path = f"{CKPT_DIR}/round_{round_idx:03d}"
    model.load_state_dict(torch.load(f"{path}/{CKPT_MODEL}"))
    optimizer.load_state_dict(torch.load(f"{path}/{CKPT_OPT}"))
    with open(f"{path}/{CKPT_META}") as f:
        meta = dict(line.strip().split("=", 1) for line in f if "=" in line)
    return meta


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="Pure PyTorch multi-round resume test")
    parser.add_argument("--rounds", type=int, default=3,
                        help="Number of save/load rounds (default: 3)")
    parser.add_argument("--epochs_per_round", type=int, default=10,
                        help="Epochs per round (default: 10)")
    args = parser.parse_args()

    N_ROUNDS = args.rounds
    EPOCHS_PER_ROUND = args.epochs_per_round

    print("=" * 70)
    print("Pure PyTorch Multi-Round Save/Load Resume Test")
    print("=" * 70)
    print(f"  Rounds: {N_ROUNDS}, epochs/round: {EPOCHS_PER_ROUND}")
    print(f"  Total epochs: {N_ROUNDS * EPOCHS_PER_ROUND}")
    print(f"  Model: nn.Embedding(user={N_USERS}+1, movie={N_MOVIES}+1, dim={EMBEDDING_DIM})")
    print(f"         + Linear({2*EMBEDDING_DIM}, 1)")

    # ── Load data ──────────────────────────────────────────────────────
    print("\nLoading data...")
    train_dataset = ML1M_Dataset(DATA_PATH, train=True)
    val_dataset = ML1M_Dataset(DATA_PATH, train=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, collate_fn=collate_fn)
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    print()

    os.makedirs(CKPT_DIR, exist_ok=True)

    all_aucs = []

    for round_idx in range(N_ROUNDS):
        print("=" * 70)
        print(f"Round {round_idx + 1}/{N_ROUNDS}")
        print("=" * 70)

        # ── Create or load model ───────────────────────────────────────
        if round_idx == 0:
            # Round 1: create fresh model
            print("  Creating fresh model...")
            model = PureTorchModel(N_USERS, N_MOVIES, EMBEDDING_DIM)
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            starting_auc = None
        else:
            # Round 2+: load from previous checkpoint
            print(f"  Loading checkpoint from round {round_idx}...")
            model = PureTorchModel(N_USERS, N_MOVIES, EMBEDDING_DIM)
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            meta = load_checkpoint(round_idx - 1, model, optimizer)
            starting_auc = float(meta["auc"])
            print(f"    Previous: epoch={meta['epoch']}, loss={meta['loss']}, auc={meta['auc']}")

        # ── Train ──────────────────────────────────────────────────────
        round_aucs = []
        for epoch in range(1, EPOCHS_PER_ROUND + 1):
            t0 = time.time()
            loss = train_epoch(model, train_loader, optimizer)
            auc = evaluate(model, val_loader)
            round_aucs.append(auc)
            all_aucs.append(auc)
            print(f"  Epoch {(round_idx * EPOCHS_PER_ROUND) + epoch:2d}: "
                  f"loss={loss:.4f}, val_auc={auc:.4f} ({time.time()-t0:.1f}s)")

        best = max(round_aucs)
        last = round_aucs[-1]
        print(f"  Round {round_idx + 1} results: best={best:.4f}, final={last:.4f}")

        if starting_auc is not None:
            print(f"  A AUC from load point: {last - starting_auc:+.4f}")

        # ── Save checkpoint ────────────────────────────────────────────
        save_checkpoint(round_idx, EPOCHS_PER_ROUND, model, optimizer, loss, last)
        print(f"  Checkpoint saved to {CKPT_DIR}/round_{round_idx:03d}/")

        # ── Summary after each round ───────────────────────────────────
        print()

    # ══════════════════════════════════════════════════════════════════
    # Final Summary
    # ══════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("Final Summary")
    print("=" * 70)
    for r in range(N_ROUNDS):
        path = f"{CKPT_DIR}/round_{r:03d}/{CKPT_META}"
        if os.path.exists(path):
            with open(path) as f:
                meta = dict(line.strip().split("=", 1) for line in f if "=" in line)
            print(f"  Round {r+1}: loss={meta['loss']}, auc={meta['auc']}")

    print(f"\n  Total epochs: {len(all_aucs)}")
    print(f"  Overall best AUC: {max(all_aucs):.4f}")

    # Multi-round trend
    first_auc = all_aucs[0] if all_aucs else 0
    last_auc = all_aucs[-1] if all_aucs else 0
    print(f"  First epoch AUC: {first_auc:.4f} → Last epoch AUC: {last_auc:.4f}")
    print(f"  Total improvement: {last_auc - first_auc:+.4f}")

    # Check if AUC keeps rising across save/load boundaries
    print("\n  AUC trend across rounds:")
    round_finals = []
    for r in range(N_ROUNDS):
        path = f"{CKPT_DIR}/round_{r:03d}/{CKPT_META}"
        if os.path.exists(path):
            with open(path) as f:
                meta = dict(line.strip().split("=", 1) for line in f if "=" in line)
            round_finals.append(float(meta["auc"]))
    if len(round_finals) >= 2:
        for i in range(len(round_finals) - 1):
            delta = round_finals[i + 1] - round_finals[i]
            direction = "rises" if delta > 0 else "drops"
            print(f"    Round {i+1} → Round {i+2}: {round_finals[i]:.4f} → "
                  f"{round_finals[i+1]:.4f} ({direction} by {abs(delta):.4f})")

    # Cleanup
    print(f"\n  Checkpoints saved in: {CKPT_DIR}")
    print("  Run 'rm -rf /tmp/pure_torch_multiround' to clean up.")


if __name__ == "__main__":
    main()
