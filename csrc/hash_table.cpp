#include "hash_table.h"
#include <cstring>
#include <stdexcept>
#include <unordered_map>

namespace hashemb {

// ---------------------------------------------------------------------------
// Bucket
// ---------------------------------------------------------------------------

void Bucket::allocate(int32_t cap) {
  capacity = cap;
  keys = new int64_t[cap];
  slot_indices = new int32_t[cap];
  dists = new int32_t[cap];
  std::memset(keys, 0xFF, sizeof(int64_t) * cap);   // -1 = empty
  std::memset(dists, 0, sizeof(int32_t) * cap);
}

bool Bucket::insert(int64_t key, int32_t slot_idx) {
  if (size >= capacity) return false;

  int32_t mask = capacity - 1;
  int32_t home = static_cast<int32_t>(key) & mask;
  int32_t idx = home;
  int32_t probe_dist = 0;

  int64_t curr_key = key;
  int32_t curr_slot = slot_idx;
  int32_t curr_dist = 0;

  while (true) {
    if (keys[idx] == -1) {
      // Empty slot — place here.
      keys[idx] = curr_key;
      slot_indices[idx] = curr_slot;
      dists[idx] = curr_dist;
      ++size;
      return true;
    }

    // Robin Hood: if current entry is closer to home, swap.
    if (dists[idx] < curr_dist) {
      // Swap.
      std::swap(keys[idx], curr_key);
      std::swap(slot_indices[idx], curr_slot);
      std::swap(dists[idx], curr_dist);
    }

    ++probe_dist;
    ++curr_dist;
    idx = (home + probe_dist) & mask;
  }
}

void Bucket::grow() {
  int32_t new_cap = capacity * 2;
  auto* new_keys = new int64_t[new_cap];
  auto* new_slots = new int32_t[new_cap];
  auto* new_dists = new int32_t[new_cap];
  std::memset(new_keys, 0xFF, sizeof(int64_t) * new_cap);
  std::memset(new_dists, 0, sizeof(int32_t) * new_cap);

  int32_t old_cap = capacity;
  int32_t new_mask = new_cap - 1;

  for (int32_t i = 0; i < old_cap; ++i) {
    if (keys[i] == -1) continue;

    int64_t key = keys[i];
    int32_t slot = slot_indices[i];
    int32_t home = static_cast<int32_t>(key) & new_mask;
    int32_t idx = home;
    int32_t probe_dist = 0;

    while (true) {
      if (new_keys[idx] == -1) {
        new_keys[idx] = key;
        new_slots[idx] = slot;
        new_dists[idx] = probe_dist;
        break;
      }
      if (new_dists[idx] < probe_dist) {
        std::swap(new_keys[idx], key);
        std::swap(new_slots[idx], slot);
        std::swap(new_dists[idx], probe_dist);
      }
      ++probe_dist;
      idx = (home + probe_dist) & new_mask;
    }
  }

  delete[] keys;
  delete[] slot_indices;
  delete[] dists;
  keys = new_keys;
  slot_indices = new_slots;
  dists = new_dists;
  capacity = new_cap;
  // size stays the same
}

int32_t* Bucket::find(int64_t key) {
  int32_t mask = capacity - 1;
  int32_t home = static_cast<int32_t>(key) & mask;
  int32_t idx = home;
  int32_t probe_dist = 0;

  while (true) {
    if (keys[idx] == -1) return nullptr;
    if (dists[idx] < static_cast<int32_t>(probe_dist)) return nullptr;
    if (keys[idx] == key) return &slot_indices[idx];

    ++probe_dist;
    idx = (home + probe_dist) & mask;
  }
}

// ---------------------------------------------------------------------------
// HashTable
// ---------------------------------------------------------------------------

HashTable::HashTable(int64_t initial_capacity_hint) {
  // Per-bucket capacity = ceil(initial_capacity_hint / 16), rounded to power-of-2.
  int64_t per_bucket = (initial_capacity_hint + kNumBuckets - 1) / kNumBuckets;
  // Add 25% slack to keep load factor ≤ 0.8.
  per_bucket = static_cast<int64_t>(per_bucket * 1.25) + 1;
  per_bucket = next_pow2(static_cast<int32_t>(per_bucket));

  for (int i = 0; i < kNumBuckets; ++i) {
    buckets_[i].allocate(static_cast<int32_t>(per_bucket));
  }
}

int64_t HashTable::find_or_create(const int64_t* keys, int32_t* slot_indices, int64_t n) {
  int64_t new_count = 0;

  // Batch-local map to handle duplicate keys within the same call.
  std::unordered_map<int64_t, int32_t> batch_map;
  batch_map.reserve(n);

  for (int64_t i = 0; i < n; ++i) {
    int64_t key = keys[i];
    if (key < 0) {
      slot_indices[i] = -1;  // sentinel for padding
      continue;
    }

    // Already resolved in this batch?
    {
      auto it = batch_map.find(key);
      if (it != batch_map.end()) {
        slot_indices[i] = it->second;
        continue;
      }
    }

    int b = static_cast<int>(key & 0xF);

    // Try shared (read) lock first.
    {
      std::shared_lock lock(buckets_[b].mtx);
      int32_t* found = buckets_[b].find(key);
      if (found) {
        slot_indices[i] = *found;
        batch_map[key] = *found;
        continue;
      }
    }

    // Need to insert — allocate a new slot.
    int32_t new_slot = static_cast<int32_t>(
        num_entries_.fetch_add(1, std::memory_order_acq_rel));

    // Exclusive (write) lock for insertion.
    {
      std::unique_lock lock(buckets_[b].mtx);
      // Double-check: another thread may have inserted this key between
      // our shared_lock release and unique_lock acquisition.
      int32_t* found = buckets_[b].find(key);
      if (found) {
        slot_indices[i] = *found;
        batch_map[key] = *found;
        num_entries_.fetch_sub(1, std::memory_order_acq_rel);  // undo slot allocation
        continue;
      }

      // Insert into bucket — auto-grow if full and retry.
      while (!buckets_[b].insert(key, new_slot)) {
        buckets_[b].grow();
      }
      slot_indices[i] = new_slot;
      batch_map[key] = new_slot;
    }
    ++new_count;
  }

  return new_count;
}

std::vector<std::pair<int64_t, int32_t>> HashTable::dump() const {
  std::vector<std::pair<int64_t, int32_t>> result;
  result.reserve(num_entries_.load(std::memory_order_relaxed));
  for (int b = 0; b < kNumBuckets; ++b) {
    std::shared_lock lock(buckets_[b].mtx);
    for (int32_t i = 0; i < buckets_[b].capacity; ++i) {
      if (buckets_[b].keys[i] != -1) {
        result.emplace_back(buckets_[b].keys[i], buckets_[b].slot_indices[i]);
      }
    }
  }
  return result;
}

int64_t HashTable::bulk_insert(const int64_t* keys, const int32_t* slots,
                                int64_t n) {
  int64_t inserted = 0;
  for (int64_t i = 0; i < n; ++i) {
    int64_t key = keys[i];
    int32_t slot = slots[i];
    int b = static_cast<int>(key & 0xF);
    std::unique_lock lock(buckets_[b].mtx);
    // Only insert if key doesn't already exist (idempotent load).
    int32_t* found = buckets_[b].find(key);
    if (found) continue;
    // Auto-grow if full and retry.
    while (!buckets_[b].insert(key, slot)) {
      buckets_[b].grow();
    }
    ++inserted;
  }
  num_entries_.fetch_add(inserted, std::memory_order_acq_rel);
  return inserted;
}

}  // namespace hashemb
