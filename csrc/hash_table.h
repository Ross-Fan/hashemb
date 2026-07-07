#pragma once

#include <atomic>
#include <cstdint>
#include <shared_mutex>
#include <cstring>
#include <vector>
#include <utility>

namespace hashemb {

constexpr int kNumBuckets = 16;

/// Per-bucket data for Robin Hood open-addressing hash map.
struct Bucket {
  std::vector<int64_t> keys;           // [capacity]; -1 = empty
  std::vector<int32_t> slot_indices;   // [capacity]
  std::vector<int32_t> dists;          // [capacity]; probe distance from home
  int32_t capacity = 0;                // allocated capacity
  int32_t size = 0;                    // live entries
  mutable std::shared_mutex mtx;

  Bucket() = default;
  Bucket(const Bucket&) = delete;
  Bucket& operator=(const Bucket&) = delete;

  // No destructor needed — vectors handle cleanup automatically.

  void allocate(int32_t cap);
  bool insert(int64_t key, int32_t slot_idx);
  int32_t* find(int64_t key);
  void grow();  // double capacity and rehash all entries
};

/// 16-way sharded hash map: feat_id → slot_index.
/// Buckets auto-grow when full; no hard capacity limit.
class HashTable {
 public:
  explicit HashTable(int64_t initial_capacity_hint);
  ~HashTable() = default;

  HashTable(const HashTable&) = delete;
  HashTable& operator=(const HashTable&) = delete;

  int64_t find_or_create(const int64_t* keys, int32_t* slot_indices, int64_t n);

  /// Get all (key, slot_index) pairs currently stored.
  std::vector<std::pair<int64_t, int32_t>> dump() const;

  /// Get entries in a single bucket (avoids allocating a vector for all 16 buckets).
  std::vector<std::pair<int64_t, int32_t>> dump_bucket(int bucket_idx) const;

  /// Bulk-insert pre-existing entries (used when loading a checkpoint).
  /// Keys must NOT already exist.  Returns number of entries inserted.
  int64_t bulk_insert(const int64_t* keys, const int32_t* slots, int64_t n);

  int64_t num_entries() const { return num_entries_.load(std::memory_order_relaxed); }

 private:
  Bucket buckets_[kNumBuckets];
  std::atomic<int64_t> num_entries_{0};
};

inline int32_t next_pow2(int32_t x) {
  if (x <= 1) return 2;
  int32_t v = x - 1;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  return v + 1;
}

}  // namespace hashemb
