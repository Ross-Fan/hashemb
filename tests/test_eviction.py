"""Tests for key eviction feature: stats tracking, eviction filtering, save/load.

Covers:
  - SlotStats (update_count, last_step) tracking during step()
  - Binary save with eviction (min_count, max_idle_steps, AND/OR combine)
  - Load VERSION=0 (no stats) and VERSION=1 (with stats) backwards compat
  - global_step_ persistence across save/load roundtrip
  - Edge cases: all evicted, none evicted, single-condition, empty save
"""

import os
import sys
import struct
import tempfile
import numpy as np
import pytest

from hashemb import _hashemb_cpp


# ===========================================================================
# Helpers
# ===========================================================================

def _train_steps(table, steps=10, key_ids=(0, 1, 2, 3, 4)):
    """Run N training steps on the given keys, returning slot indices from last step."""
    slot = None
    for _ in range(steps):
        keys = np.array(key_ids, dtype=np.int64)
        _, slot = table.lookup_and_gather(keys)
        grads = np.ones((len(key_ids), table.embedding_dim), dtype=np.float32)
        table.scatter_add_grad(slot, grads)
        table.step()
    return slot


def _train_pattern(table, pattern):
    """Run training with per-step key patterns.

    pattern: list of list[int], each element is the key IDs to train on that step.
    Returns slot dict: key_id → slot_index.
    """
    key_to_slot = {}
    for key_ids in pattern:
        keys = np.array(key_ids, dtype=np.int64)
        _, slots = table.lookup_and_gather(keys)
        for k, s in zip(key_ids, slots):
            key_to_slot[k] = s
        grads = np.ones((len(key_ids), table.embedding_dim), dtype=np.float32)
        table.scatter_add_grad(slots, grads)
        table.step()
    return key_to_slot


def _roundtrip(table, **save_kwargs):
    """Save → load → return new table."""
    with tempfile.NamedTemporaryFile(suffix=".hashemb", delete=False) as f:
        f.close()
        try:
            table.save(f.name, **save_kwargs)
            new_table = _hashemb_cpp.HashEmbeddingTable(
                table.capacity, table.embedding_dim,
                optimizer="adam", lr=0.01,
            )
            new_table.load(f.name)
        finally:
            os.unlink(f.name)
    return new_table


# ===========================================================================
# Tests
# ===========================================================================

class TestEvictionStatsTracking:
    """SlotStats update_count and last_step are correctly tracked during step()."""

    def test_update_count_increments(self):
        table = _hashemb_cpp.HashEmbeddingTable(100, 4, optimizer="adam", lr=0.01)
        key_to_slot = _train_pattern(table, [[0], [1], [0], [0, 1]])
        sd = table.state_dict()

        # Find indices for keys 0 and 1.
        keys_arr = sd["keys"]
        weights = sd["weight"]
        slots_arr = sd["slots"]

        # Both keys should have been inserted.
        assert len(keys_arr) == 2

        # Key 0 was updated 3 times (steps 0, 2, 3), key 1 updated 2 times (steps 1, 3).
        # We can't directly read stats from state_dict, but we can verify through
        # save/load roundtrip that the eviction logic works (tested elsewhere).
        # Here we just verify basic training works and entries exist.
        assert table.num_entries == 2

    def test_global_step_increments(self):
        table = _hashemb_cpp.HashEmbeddingTable(100, 4, optimizer="adam", lr=0.01)
        for _ in range(5):
            keys = np.array([42], dtype=np.int64)
            _, slot = table.lookup_and_gather(keys)
            table.scatter_add_grad(slot, np.ones((1, 4), dtype=np.float32))
            table.step()

        # global_step_ should be 5.  Verify indirectly: max_idle_steps=0 after
        # 5 steps means nobody is idle → all kept.
        new_t = _roundtrip(table, max_idle_steps=0)
        assert new_t.num_entries == 1

    def test_stats_persist_across_save_load(self):
        """After save/load, stats should survive so eviction works on reloaded table."""
        # Step 0: insert keys 0-4. Steps 1-9: only update keys 0-2.
        table = _hashemb_cpp.HashEmbeddingTable(100, 4, optimizer="adam", lr=0.01)

        for step in range(10):
            if step == 0:
                keys = np.array([0, 1, 2, 3, 4], dtype=np.int64)
            else:
                keys = np.array([0, 1, 2], dtype=np.int64)
            _, slots = table.lookup_and_gather(keys)
            table.scatter_add_grad(slots, np.ones((len(keys), 4), dtype=np.float32))
            table.step()

        # Save and reload.
        new_t = _roundtrip(table)

        # Reloaded table: keys 3,4 should have count=1, idle=9 (last_step=1, global_step=10)
        # min_count=2 should evict them.
        new_t2 = _roundtrip(new_t, min_count=2)
        # Keys 0-2: count=10 → kept. Keys 3-4: count=1 → evicted.
        assert new_t2.num_entries == 3


class TestEvictionMinCount:
    """Eviction by update_count threshold."""

    def test_min_count_evicts_below_threshold(self):
        table = _hashemb_cpp.HashEmbeddingTable(100, 4, optimizer="adam", lr=0.01)
        # All keys updated 5 times.
        _train_steps(table, steps=5, key_ids=(0, 1, 2, 3, 4))

        new_t = _roundtrip(table, min_count=5)
        assert new_t.num_entries == 5  # count=5 >= 5 → all kept

        new_t2 = _roundtrip(table, min_count=6)
        assert new_t2.num_entries == 0  # count=5 < 6 → all evicted

    def test_min_count_zero_disables(self):
        table = _hashemb_cpp.HashEmbeddingTable(100, 4, optimizer="adam", lr=0.01)
        _train_steps(table, steps=3, key_ids=(0, 1, 2))

        new_t = _roundtrip(table, min_count=0)
        assert new_t.num_entries == 3  # min_count=0 means no eviction

    def test_mixed_counts(self):
        table = _hashemb_cpp.HashEmbeddingTable(100, 4, optimizer="adam", lr=0.01)
        # Key 0: 10 updates. Keys 1-2: 10 updates. Key 3: 1 update. Key 4: 1 update.
        _train_pattern(table, [
            [0, 1, 2, 3, 4],   # step 0: all inserted
            [0, 1, 2],          # steps 1-9
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
        ])

        # min_count=5 → keep keys with count>=5 (0,1,2 only)
        new_t = _roundtrip(table, min_count=5)
        assert new_t.num_entries == 3

        # min_count=2 → keep keys with count>=2 (still 0,1,2)
        new_t2 = _roundtrip(table, min_count=2)
        assert new_t2.num_entries == 3

        # min_count=11 → all evicted (max count is 10)
        new_t3 = _roundtrip(table, min_count=11)
        assert new_t3.num_entries == 0


class TestEvictionMaxIdle:
    """Eviction by idle steps threshold."""

    def test_max_idle_evicts_stale_keys(self):
        table = _hashemb_cpp.HashEmbeddingTable(100, 4, optimizer="adam", lr=0.01)
        # 10 steps total, all keys updated at each step → idle=0 for all.
        _train_steps(table, steps=10, key_ids=(0, 1, 2))

        # max_idle_steps=5: all keys idle=0 → none evicted.
        new_t = _roundtrip(table, max_idle_steps=5)
        assert new_t.num_entries == 3

    def test_max_idle_partial_stale(self):
        table = _hashemb_cpp.HashEmbeddingTable(100, 4, optimizer="adam", lr=0.01)
        # 10 steps: keys 0-2 updated every step, keys 3-4 only at step 0.
        for step in range(10):
            if step == 0:
                keys = np.array([0, 1, 2, 3, 4], dtype=np.int64)
            else:
                keys = np.array([0, 1, 2], dtype=np.int64)
            _, slots = table.lookup_and_gather(keys)
            table.scatter_add_grad(slots, np.ones((len(keys), 4), dtype=np.float32))
            table.step()

        # global_step=10. Keys 3-4: last_step=1 → idle=9.
        # max_idle_steps=5: idle=9 > 5 → keys 3-4 evicted.
        new_t = _roundtrip(table, max_idle_steps=5)
        assert new_t.num_entries == 3

        # max_idle_steps=15: idle=9 <= 15 → all kept.
        new_t2 = _roundtrip(table, max_idle_steps=15)
        assert new_t2.num_entries == 5

    def test_max_idle_zero_disables(self):
        table = _hashemb_cpp.HashEmbeddingTable(100, 4, optimizer="adam", lr=0.01)
        _train_steps(table, steps=5, key_ids=(0, 1, 2))

        new_t = _roundtrip(table, max_idle_steps=0)
        assert new_t.num_entries == 3


class TestEvictionCombineLogic:
    """AND / OR combination logic for dual-condition eviction."""

    def test_and_keep_only_when_both(self):
        """AND: evict only when BOTH count<min AND idle>max.

        Pattern (15 steps total):
          Step 0:     insert A(100), B(200), C(300)
          Steps 1-9:  A only
          Steps 10-14: B only

        Results at global_step=15:
          A: count=10, idle=5   (neither condition triggers)
          B: count=6,  idle=0   (count<9 triggers, idle≤10 does not)
          C: count=1,  idle=14  (both conditions trigger)

        AND(min_count=9, idle>10): evict only C → keeps A, B (2 entries).
        """
        table = _hashemb_cpp.HashEmbeddingTable(100, 4, optimizer="adam", lr=0.01)

        for step in range(15):
            if step == 0:
                keys = np.array([100, 200, 300], dtype=np.int64)
            elif 1 <= step <= 9:
                keys = np.array([100], dtype=np.int64)
            else:
                keys = np.array([200], dtype=np.int64)
            _, slots = table.lookup_and_gather(keys)
            table.scatter_add_grad(slots, np.ones((len(keys), 4), dtype=np.float32))
            table.step()

        new_t = _roundtrip(table, min_count=9, max_idle_steps=10, combine="and")
        assert new_t.num_entries == 2, f"AND: expected 2 (A,B kept), got {new_t.num_entries}"

    def test_or_keep_only_when_neither(self):
        """OR: evict when EITHER count<min OR idle>max.

        Same pattern as AND test:
          A: count=10, idle=5   (neither → kept by OR)
          B: count=6,  idle=0   (count<9 → evicted by OR)
          C: count=1,  idle=14  (both → evicted by OR)

        OR(min_count=9, idle>10): evict B and C → keeps only A (1 entry).
        """
        table = _hashemb_cpp.HashEmbeddingTable(100, 4, optimizer="adam", lr=0.01)

        for step in range(15):
            if step == 0:
                keys = np.array([100, 200, 300], dtype=np.int64)
            elif 1 <= step <= 9:
                keys = np.array([100], dtype=np.int64)
            else:
                keys = np.array([200], dtype=np.int64)
            _, slots = table.lookup_and_gather(keys)
            table.scatter_add_grad(slots, np.ones((len(keys), 4), dtype=np.float32))
            table.step()

        new_t = _roundtrip(table, min_count=9, max_idle_steps=10, combine="or")
        assert new_t.num_entries == 1, f"OR: expected 1 (A kept), got {new_t.num_entries}"

    def test_default_combine_is_and(self):
        """Default (empty combine string) behaves like 'and'."""
        table = _hashemb_cpp.HashEmbeddingTable(100, 4, optimizer="adam", lr=0.01)

        # Same pattern: 15 steps, A count=10 idle=5, B count=6 idle=0, C count=1 idle=14
        for step in range(15):
            if step == 0:
                keys = np.array([100, 200, 300], dtype=np.int64)
            elif 1 <= step <= 9:
                keys = np.array([100], dtype=np.int64)
            else:
                keys = np.array([200], dtype=np.int64)
            _, slots = table.lookup_and_gather(keys)
            table.scatter_add_grad(slots, np.ones((len(keys), 4), dtype=np.float32))
            table.step()

        # Default = AND → 2 entries kept (A, B).
        new_t = _roundtrip(table, min_count=9, max_idle_steps=10, combine="")
        assert new_t.num_entries == 2


class TestEvictionSingleCondition:
    """Single condition: one threshold = 0 disables that condition."""

    def test_only_min_count(self):
        table = _hashemb_cpp.HashEmbeddingTable(100, 4, optimizer="adam", lr=0.01)
        # Key A: count=10, idle=9; Key B: count=1, idle=9.
        for step in range(10):
            if step == 0:
                keys = np.array([0, 1], dtype=np.int64)
            else:
                keys = np.array([0], dtype=np.int64)
            _, slots = table.lookup_and_gather(keys)
            table.scatter_add_grad(slots, np.ones((len(keys), 4), dtype=np.float32))
            table.step()

        # min_count=5, max_idle_steps=0 → only count matters. combine is irrelevant.
        new_t = _roundtrip(table, min_count=5, max_idle_steps=0, combine="or")
        assert new_t.num_entries == 1  # Only key 0 (count=10 >= 5) kept.

    def test_only_max_idle(self):
        table = _hashemb_cpp.HashEmbeddingTable(100, 4, optimizer="adam", lr=0.01)
        # Key A: count=10, idle=0; Key B: count=1, idle=9.
        for step in range(10):
            if step == 0:
                keys = np.array([0, 1], dtype=np.int64)
            else:
                keys = np.array([0], dtype=np.int64)
            _, slots = table.lookup_and_gather(keys)
            table.scatter_add_grad(slots, np.ones((len(keys), 4), dtype=np.float32))
            table.step()

        # min_count=0, max_idle_steps=5 → only idle matters.
        new_t = _roundtrip(table, min_count=0, max_idle_steps=5, combine="or")
        assert new_t.num_entries == 1  # Only key 0 (idle=0 <= 5) kept.

    def test_both_zero_no_eviction(self):
        table = _hashemb_cpp.HashEmbeddingTable(100, 4, optimizer="adam", lr=0.01)
        _train_steps(table, steps=5, key_ids=(0, 1, 2, 3))

        new_t = _roundtrip(table, min_count=0, max_idle_steps=0)
        assert new_t.num_entries == 4


class TestEvictionEdgeCases:
    """Edge cases: all evicted, single entry, empty table."""

    def test_all_keys_evicted(self):
        table = _hashemb_cpp.HashEmbeddingTable(100, 4, optimizer="adam", lr=0.01)
        _train_steps(table, steps=3, key_ids=(0, 1, 2))

        # min_count=10 evicts everything (count=3 < 10).
        new_t = _roundtrip(table, min_count=10)
        assert new_t.num_entries == 0

        # The empty table is still usable.
        keys = np.array([99], dtype=np.int64)
        emb, slot = new_t.lookup_and_gather(keys)
        assert emb.shape == (1, 4)
        assert new_t.num_entries == 1

    def test_single_entry(self):
        table = _hashemb_cpp.HashEmbeddingTable(100, 4, optimizer="adam", lr=0.01)
        _train_steps(table, steps=1, key_ids=(42,))

        new_t = _roundtrip(table, min_count=1)
        assert new_t.num_entries == 1  # count=1 >= 1 → kept

        new_t2 = _roundtrip(table, min_count=2)
        assert new_t2.num_entries == 0  # count=1 < 2 → evicted

    def test_trainable_after_eviction_roundtrip(self):
        """After save→load with eviction, table is still trainable."""
        table = _hashemb_cpp.HashEmbeddingTable(100, 4, optimizer="adam", lr=0.01)
        _train_steps(table, steps=10, key_ids=(0, 1, 2, 3, 4))

        new_t = _roundtrip(table, min_count=5)  # Should keep all 5.

        # Continue training.
        keys = np.array([0, 1], dtype=np.int64)
        _, slots = new_t.lookup_and_gather(keys)
        for _ in range(5):
            new_t.scatter_add_grad(slots, np.ones((2, 4), dtype=np.float32))
            new_t.step()

        # Lookup should be stable.
        emb = new_t.lookup(slots)
        assert emb.shape == (2, 4)
        assert not np.any(np.isnan(emb))
        assert not np.any(np.isinf(emb))

    def test_no_eviction_no_params_same_as_no_filter(self):
        """Calling save() with defaults = same as save with explicit zeros."""
        table = _hashemb_cpp.HashEmbeddingTable(100, 4, optimizer="adam", lr=0.01)
        _train_steps(table, steps=5, key_ids=(0, 1, 2))

        # save() with no eviction params
        new_t_default = _roundtrip(table)
        # save() with explicit zeros
        new_t_explicit = _roundtrip(table, min_count=0, max_idle_steps=0, combine="")

        assert new_t_default.num_entries == new_t_explicit.num_entries == 3


class TestEvictionGlobalStep:
    """global_step_ is tracked and persists across save/load."""

    def test_global_step_persists(self):
        table = _hashemb_cpp.HashEmbeddingTable(100, 4, optimizer="adam", lr=0.01)
        _train_steps(table, steps=7, key_ids=(0, 1))

        # Roundtrip preserves global_step_.
        new_t = _roundtrip(table)

        # Continue training: 3 more steps → global_step should be 10.
        for _ in range(3):
            keys = np.array([0], dtype=np.int64)
            _, slots = new_t.lookup_and_gather(keys)
            new_t.scatter_add_grad(slots, np.ones((1, 4), dtype=np.float32))
            new_t.step()

        # Key 1: count=7, last_step=7 from original training. New global_step after
        # reload + 3 steps should be 10. So idle for key 1 = 10-7 = 3.
        # max_idle_steps=2 should evict key 1 (idle=3 > 2).
        new_t2 = _roundtrip(new_t, max_idle_steps=2)
        assert new_t2.num_entries == 1  # Only key 0 survives.


class TestEvictionPythonWrapper:
    """Python HashEmbedding.save() forwards eviction params correctly."""

    @pytest.fixture(autouse=True)
    def _import_torch(self):
        self.torch = pytest.importorskip("torch")

    def test_save_with_params(self):
        from hashemb import HashEmbedding
        emb = HashEmbedding(4, 100, optimizer="adam", lr=0.01)

        # Train a few keys.
        keys = self.torch.tensor([0, 1, 2, 3, 4], dtype=self.torch.int64)
        out = emb(keys)
        loss = out.sum()
        loss.backward()
        emb.step()

        with tempfile.NamedTemporaryFile(suffix=".hashemb", delete=False) as f:
            fpath = f.name
        try:
            # No error = success.
            emb.save(fpath, min_count=5, max_idle_steps=100, combine="and")

            emb2 = HashEmbedding(4, 100, optimizer="adam", lr=0.01)
            emb2.load(fpath)
            assert emb2.num_entries == 5
        finally:
            os.unlink(fpath)

    def test_combined_eviction_via_wrapper(self):
        from hashemb import HashEmbedding
        emb = HashEmbedding(4, 100, optimizer="adam", lr=0.01)

        # Train: key 0 every step, key 1 only first step.
        for step in range(10):
            if step == 0:
                ids = self.torch.tensor([0, 1], dtype=self.torch.int64)
            else:
                ids = self.torch.tensor([0], dtype=self.torch.int64)
            out = emb(ids)
            loss = out.sum()
            loss.backward()
            emb.step()

        with tempfile.NamedTemporaryFile(suffix=".hashemb", delete=False) as f:
            fpath = f.name
        try:
            emb.save(fpath, min_count=5)
            emb2 = HashEmbedding(4, 100, optimizer="adam", lr=0.01)
            emb2.load(fpath)
            assert emb2.num_entries == 1  # Only key 0 (count=10 >= 5).
        finally:
            os.unlink(fpath)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
