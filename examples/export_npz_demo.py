#!/usr/bin/env python3
"""Demo: export HashEmb embeddings to NPZ and read them back."""

import tempfile

import numpy as np
import torch

from hashemb import HashEmbedding


def main():
    emb = HashEmbedding(embedding_dim=4, capacity=100, optimizer="sgd", lr=0.1)

    # Insert and train a few hash IDs so vectors are non-zero.
    keys = torch.tensor([10, 20, 30], dtype=torch.int64)
    out = emb(keys)
    out.sum().backward()
    emb.step()

    with tempfile.NamedTemporaryFile(suffix=".npz") as f:
        count = emb.export(f.name)
        print(f"exported entries: {count}")

        z = np.load(f.name)
        exported_keys = z["keys"]              # int64[N]
        embeddings = z["embeddings"]          # float32[N, D]
        dim = int(z["dim"])
        num_entries = int(z["num_entries"])
        format_version = int(z["format_version"])

        print(f"dim: {dim}")
        print(f"num_entries: {num_entries}")
        print(f"format_version: {format_version}")
        print(f"keys dtype/shape: {exported_keys.dtype} {exported_keys.shape}")
        print(f"embeddings dtype/shape: {embeddings.dtype} {embeddings.shape}")

        # Export order is unspecified, so align vectors by hash ID.
        embedding_by_key = {
            int(key): embeddings[i]
            for i, key in enumerate(exported_keys)
        }

        target_key = 20
        print(f"embedding for key {target_key}: {embedding_by_key[target_key]}")


if __name__ == "__main__":
    main()
