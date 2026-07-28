#!/usr/bin/env python3
"""Demo: export a saved HashEmb checkpoint to bucket-sharded NPZ files."""

import tempfile
from pathlib import Path

import numpy as np
import torch

from hashemb import HashEmbedding
from hashemb.utils import export_hashemb_to_npz

NUM_BUCKETS = 16


def main():
    emb = HashEmbedding(embedding_dim=4, capacity=100, optimizer="sgd", lr=0.1)

    # Insert and train a few hash IDs so vectors are non-zero.
    keys = torch.tensor([10, 20, 30], dtype=torch.int64)
    out = emb(keys)
    out.sum().backward()
    emb.step()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save the hash table to binary checkpoint.
        save_path = Path(tmpdir) / "table.hashemb"
        emb.save(str(save_path))

        # Export to bucket-sharded NPZ files.
        npz_base = Path(tmpdir) / "embeddings.npz"
        count = export_hashemb_to_npz(
            str(save_path), str(npz_base),
            embedding_dim=4, capacity=100,
        )
        print(f"exported entries: {count}")

        # Read back all bucket files, aligning vectors by hash ID.
        embedding_by_key = {}
        total = 0

        for bucket_id in range(NUM_BUCKETS):
            bucket_path = npz_base.with_name(
                f"{npz_base.stem}_bucket_{bucket_id:02d}{npz_base.suffix}"
            )
            z = np.load(bucket_path)
            exported_keys = z["keys"]            # int64[Nb]
            embeddings = z["embeddings"]        # float32[Nb, D]
            dim = int(z["dim"])
            num_entries = int(z["num_entries"])
            format_version = int(z["format_version"])

            if num_entries:
                print(
                    f"bucket {bucket_id:02d}: entries={num_entries}, "
                    f"dim={dim}, format_version={format_version}, "
                    f"keys={exported_keys.dtype}{exported_keys.shape}, "
                    f"embeddings={embeddings.dtype}{embeddings.shape}"
                )

            total += num_entries
            for key, vector in zip(exported_keys, embeddings):
                embedding_by_key[int(key)] = vector

        print(f"loaded entries: {total}")
        target_key = 20
        print(f"embedding for key {target_key}: {embedding_by_key[target_key]}")


if __name__ == "__main__":
    main()
