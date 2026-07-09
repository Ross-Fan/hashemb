#pragma once

#include "hash_table.h"
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>

namespace hashemb {

/// Optimizer configuration.
struct OptimizerConfig {
  enum Type { SGD, ADAM };
  Type type = SGD;
  float lr = 0.01f;
  float beta1 = 0.9f;
  float beta2 = 0.999f;
  float eps = 1e-8f;
};

// ---------------------------------------------------------------------------
// Block-based storage for embedding vectors
// ---------------------------------------------------------------------------

static constexpr int64_t kDefaultBlockSize = 1'000'000;  // 1M slots per block

/// A single block of [block_size, D] float data, allocated on demand.
struct Block {
  float* data = nullptr;

  void allocate(int64_t block_size, int64_t dim) {
    size_t bytes = static_cast<size_t>(block_size) *
                   static_cast<size_t>(dim) * sizeof(float);
    void* p = nullptr;
    if (posix_memalign(&p, 64, bytes) != 0) throw std::bad_alloc();
    data = static_cast<float*>(p);
    std::memset(data, 0, bytes);
  }

  void deallocate() {
    if (data) {
      std::free(data);
      data = nullptr;
    }
  }
};

/// Get pointer to slot `slot_id` within a block vector using given block_size.
inline float* slot_ptr(const std::vector<Block>& blocks, int64_t slot_id, int32_t D, int64_t block_size) {
  int64_t block_id = slot_id / block_size;
  int64_t offset = slot_id % block_size;
  return blocks[block_id].data + offset * D;
}

/// Host-memory embedding table with block-based storage, gradient
/// accumulation and native optimizers.
///
/// Memory layout (block chain, allocated on demand):
///   emb_blocks_   [n_blocks][block_size, D]  ← embedding vectors
///   grad_blocks_  [n_blocks][block_size, D]  ← accumulated gradients
///   m_blocks_     [n_blocks][block_size, D]  ← Adam 1st moment (ADAM only)
///   v_blocks_     [n_blocks][block_size, D]  ← Adam 2nd moment (ADAM only)
///   t_                                       ← Adam timestep
///
/// Data flow:
///   scatter_add_grad(slots, grads)    — called by backward(), accumulates into grad_blocks_
///   step()                           — applies grad via SGD/Adam, then zeroes grad
///   zero_grad()                      — clears grad_blocks_
class EmbeddingTable {
 public:
  EmbeddingTable(int64_t initial_capacity, int32_t embedding_dim,
                 const OptimizerConfig& opt_cfg = OptimizerConfig{},
                 int64_t block_size = kDefaultBlockSize,
                 float initial_scale = 0.0f);
  ~EmbeddingTable();

  EmbeddingTable(const EmbeddingTable&) = delete;
  EmbeddingTable& operator=(const EmbeddingTable&) = delete;

  // ── Lookup ──────────────────────────────────────────────────────────

  void find_or_create(const int64_t* keys, int32_t* slot_indices, int64_t n) {
    hash_table_.find_or_create(keys, slot_indices, n);
  }

  void lookup(const int32_t* slot_indices, float* output, int64_t n) const;

  void lookup_and_gather(const int64_t* keys, float* output,
                         int32_t* slot_indices, int64_t n);

  // ── Gradient accumulation (called by autograd backward) ─────────────

  void scatter_add_grad(const int32_t* slot_indices, const float* grads,
                        int64_t n);

  // ── Optimizer step ──────────────────────────────────────────────────

  void step();
  void zero_grad();

  // ── Serialisation ───────────────────────────────────────────────────

  /// Save to binary file (bucket-by-bucket, zero extra memory allocation).
  void save(const std::string& path) const;

  /// Load from binary file written by save().
  void load(const std::string& path);

  /// state_dict fields (pybind11 bridge expects numpy-compatible layout):
  ///   keys:       int64[ num_entries ]  — feat_id for each occupied slot
  ///   slots:      int32[ num_entries ]  — slot index
  ///   weight:     float32[ num_entries, D ]  — embedding values
  ///   grad:       float32[ num_entries, D ]  — accumulated gradients
  ///   opt_type:   "sgd" | "adam"
  ///   lr, beta1, beta2, eps: float
  ///   t:          int64  — Adam timestep (only if Adam)
  ///   m:          float32[ num_entries, D ]  — Adam m (only if Adam)
  ///   v:          float32[ num_entries, D ]  — Adam v (only if Adam)
  ///   dim:        int32
  void state_dict_arrays(
      /* out */ std::vector<int64_t>& keys,
      /* out */ std::vector<int32_t>& slots,
      /* out */ std::vector<float>& weight,
      /* out */ std::vector<float>& grad,
      /* out */ std::vector<float>& m,
      /* out */ std::vector<float>& v,
      /* out */ int64_t& t,
      /* out */ std::string& opt_type_str) const;

  /// Restore from a previously exported state.
  void load_state_dict_arrays(
      int64_t n,
      const int64_t* keys, const int32_t* slots,
      const float* weight, const float* grad,
      const float* m, const float* v,
      int64_t t, const std::string& opt_type_str);

  // ── Accessors ───────────────────────────────────────────────────────

  int64_t initial_capacity() const { return initial_capacity_; }
  int32_t embedding_dim() const { return embedding_dim_; }
  int64_t num_entries() const { return hash_table_.num_entries(); }
  const OptimizerConfig& opt_config() const { return opt_cfg_; }

 private:
  /// Ensure blocks exist for the given slot_id (allocate if needed).
  void ensure_slot(int64_t slot_id);

  int64_t initial_capacity_;
  int32_t embedding_dim_;
  int64_t block_size_;
  float initial_scale_ = 0.0f;
  OptimizerConfig opt_cfg_;

  // Block-based storage (allocated on demand via ensure_slot)
  std::vector<Block> emb_blocks_;
  std::vector<Block> grad_blocks_;

  // Adam state (empty for SGD)
  std::vector<Block> m_blocks_;
  std::vector<Block> v_blocks_;
  int64_t t_ = 0;

  // Dirty-slot tracking for sparse step().
  // scatter_add_grad marks slots as dirty; step() only iterates dirty_slots_.
  // NOT std::vector<bool> — the packed bitset causes data races when writing
  // different bools from parallel threads (adjacent bits share the same byte).
  std::vector<uint8_t> slot_dirty_;
  std::vector<int32_t> dirty_slots_;

  HashTable hash_table_;
};

}  // namespace hashemb
