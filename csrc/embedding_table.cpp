#include "embedding_table.h"
#include <cstring>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <utility>

namespace hashemb {

// ===========================================================================
// Construction / destruction
// ===========================================================================

EmbeddingTable::EmbeddingTable(int64_t initial_capacity, int32_t embedding_dim,
                               const OptimizerConfig& opt_cfg)
    : initial_capacity_(initial_capacity),
      embedding_dim_(embedding_dim),
      opt_cfg_(opt_cfg),
      hash_table_(initial_capacity) {
  if (initial_capacity <= 0 || embedding_dim <= 0) {
    throw std::invalid_argument("capacity and embedding_dim must be positive");
  }
  // No pre-allocation of embedding buffers — blocks are allocated on demand
  // via ensure_slot() during lookup_and_gather() or load_state_dict_arrays().
}

EmbeddingTable::~EmbeddingTable() {
  for (auto& b : emb_blocks_) b.deallocate();
  for (auto& b : grad_blocks_) b.deallocate();
  for (auto& b : m_blocks_) b.deallocate();
  for (auto& b : v_blocks_) b.deallocate();
}

// ===========================================================================
// Block management
// ===========================================================================

void EmbeddingTable::ensure_slot(int64_t slot_id) {
  int64_t needed = (slot_id / kBlockSize) + 1;
  while (static_cast<int64_t>(emb_blocks_.size()) < needed) {
    emb_blocks_.emplace_back();
    emb_blocks_.back().allocate(embedding_dim_);
    grad_blocks_.emplace_back();
    grad_blocks_.back().allocate(embedding_dim_);
    if (opt_cfg_.type == OptimizerConfig::ADAM) {
      m_blocks_.emplace_back();
      m_blocks_.back().allocate(embedding_dim_);
      v_blocks_.emplace_back();
      v_blocks_.back().allocate(embedding_dim_);
    }
  }
}

// ===========================================================================
// Lookup
// ===========================================================================

void EmbeddingTable::lookup(const int32_t* slot_indices, float* output,
                            int64_t n) const {
  int32_t D = embedding_dim_;
  for (int64_t i = 0; i < n; ++i) {
    int32_t slot = slot_indices[i];
    if (slot < 0) {
      std::memset(output + i * D, 0, sizeof(float) * D);
    } else {
      std::memcpy(output + i * D, slot_ptr(emb_blocks_, slot, D),
                  sizeof(float) * D);
    }
  }
}

void EmbeddingTable::lookup_and_gather(const int64_t* keys, float* output,
                                       int32_t* slot_indices, int64_t n) {
  hash_table_.find_or_create(keys, slot_indices, n);

  // Ensure backing blocks for the maximum slot ID assigned.
  int32_t max_slot = -1;
  for (int64_t i = 0; i < n; ++i) {
    if (slot_indices[i] > max_slot) max_slot = slot_indices[i];
  }
  if (max_slot >= 0) ensure_slot(max_slot);

  lookup(slot_indices, output, n);
}

// ===========================================================================
// Gradient accumulation
// ===========================================================================

void EmbeddingTable::scatter_add_grad(const int32_t* slot_indices,
                                      const float* grads,
                                      int64_t n) {
  int32_t D = embedding_dim_;

  // Ensure blocks for all referenced slots.
  int32_t max_slot = -1;
  for (int64_t i = 0; i < n; ++i) {
    if (slot_indices[i] > max_slot) max_slot = slot_indices[i];
  }
  if (max_slot >= 0) ensure_slot(max_slot);

  // Sort indices by slot to group duplicates.
  auto* idx = new int64_t[n];
  for (int64_t i = 0; i < n; ++i) idx[i] = i;
  std::sort(idx, idx + n, [&](int64_t a, int64_t b) {
    return slot_indices[a] < slot_indices[b];
  });

  int64_t i = 0;
  while (i < n) {
    int32_t slot = slot_indices[idx[i]];
    if (slot < 0) { ++i; continue; }
    int64_t j = i;
    while (j < n && slot_indices[idx[j]] == slot) ++j;

    float* grad_dst = slot_ptr(grad_blocks_, slot, D);
    for (int64_t k = i; k < j; ++k) {
      const float* g = grads + idx[k] * D;
      for (int32_t d = 0; d < D; ++d) grad_dst[d] += g[d];
    }
    i = j;
  }
  delete[] idx;
}

// ===========================================================================
// Optimizer step
// ===========================================================================

void EmbeddingTable::zero_grad() {
  int32_t D = embedding_dim_;
  for (auto& block : grad_blocks_) {
    if (block.data) {
      std::memset(block.data, 0, static_cast<size_t>(kBlockSize) * D * sizeof(float));
    }
  }
}

void EmbeddingTable::step() {
  int32_t D = embedding_dim_;
  int64_t n = num_entries();
  if (n == 0) return;

  // Ensure all blocks up to the last slot exist.
  ensure_slot(n - 1);

  if (opt_cfg_.type == OptimizerConfig::SGD) {
    float lr = opt_cfg_.lr;
    for (int64_t slot = 0; slot < n; ++slot) {
      float* w = slot_ptr(emb_blocks_, slot, D);
      float* g = slot_ptr(grad_blocks_, slot, D);
      for (int32_t d = 0; d < D; ++d) {
        w[d] -= lr * g[d];
      }
    }
  } else if (opt_cfg_.type == OptimizerConfig::ADAM) {
    ++t_;
    float lr = opt_cfg_.lr;
    float b1 = opt_cfg_.beta1;
    float b2 = opt_cfg_.beta2;
    float eps = opt_cfg_.eps;
    float bias_corr1 = 1.0f - std::pow(b1, static_cast<float>(t_));
    float bias_corr2 = 1.0f - std::pow(b2, static_cast<float>(t_));

    for (int64_t slot = 0; slot < n; ++slot) {
      float* w = slot_ptr(emb_blocks_, slot, D);
      float* g = slot_ptr(grad_blocks_, slot, D);
      float* m = slot_ptr(m_blocks_, slot, D);
      float* v = slot_ptr(v_blocks_, slot, D);

      for (int32_t d = 0; d < D; ++d) {
        m[d] = b1 * m[d] + (1.0f - b1) * g[d];
        v[d] = b2 * v[d] + (1.0f - b2) * g[d] * g[d];
        float m_hat = m[d] / bias_corr1;
        float v_hat = v[d] / bias_corr2;
        w[d] -= lr * m_hat / (std::sqrt(v_hat) + eps);
      }
    }
  }

  // Clear gradients after applying.
  zero_grad();
}

// ===========================================================================
// Serialisation
// ===========================================================================

void EmbeddingTable::state_dict_arrays(
    std::vector<int64_t>& keys,
    std::vector<int32_t>& slots,
    std::vector<float>& weight,
    std::vector<float>& grad,
    std::vector<float>& m,
    std::vector<float>& v,
    int64_t& t,
    std::string& opt_type_str) const {

  auto entries = hash_table_.dump();
  int64_t n = static_cast<int64_t>(entries.size());
  int32_t D = embedding_dim_;

  keys.resize(n);
  slots.resize(n);
  weight.resize(n * D);
  grad.resize(n * D);
  m.assign(n * D, 0.0f);
  v.assign(n * D, 0.0f);

  for (int64_t i = 0; i < n; ++i) {
    keys[i] = entries[i].first;
    int32_t slot = entries[i].second;
    slots[i] = slot;
    std::memcpy(&weight[i * D], slot_ptr(emb_blocks_, slot, D), sizeof(float) * D);
    std::memcpy(&grad[i * D],   slot_ptr(grad_blocks_, slot, D), sizeof(float) * D);
  }

  if (opt_cfg_.type == OptimizerConfig::ADAM) {
    opt_type_str = "adam";
    t = t_;
    for (int64_t i = 0; i < n; ++i) {
      int32_t slot = entries[i].second;
      std::memcpy(&m[i * D], slot_ptr(m_blocks_, slot, D), sizeof(float) * D);
      std::memcpy(&v[i * D], slot_ptr(v_blocks_, slot, D), sizeof(float) * D);
    }
  } else {
    opt_type_str = "sgd";
    t = 0;
  }
}

void EmbeddingTable::load_state_dict_arrays(
    int64_t n,
    const int64_t* keys, const int32_t* slots,
    const float* weight, const float* grad,
    const float* m, const float* v,
    int64_t t, const std::string& opt_type_str) {

  int32_t D = embedding_dim_;

  // Populate hash table.
  hash_table_.bulk_insert(keys, slots, n);

  // Ensure blocks for all referenced slots.
  if (n > 0) {
    int32_t max_slot = slots[0];
    for (int64_t i = 1; i < n; ++i) {
      if (slots[i] > max_slot) max_slot = slots[i];
    }
    ensure_slot(max_slot);
  }

  // Copy buffers.
  for (int64_t i = 0; i < n; ++i) {
    int32_t slot = slots[i];
    std::memcpy(slot_ptr(emb_blocks_, slot, D),  weight + i * D, sizeof(float) * D);
    std::memcpy(slot_ptr(grad_blocks_, slot, D), grad   + i * D, sizeof(float) * D);
  }

  if (opt_cfg_.type == OptimizerConfig::ADAM && !opt_type_str.empty()) {
    t_ = t;
    for (int64_t i = 0; i < n; ++i) {
      int32_t slot = slots[i];
      std::memcpy(slot_ptr(m_blocks_, slot, D), m + i * D, sizeof(float) * D);
      std::memcpy(slot_ptr(v_blocks_, slot, D), v + i * D, sizeof(float) * D);
    }
  }
}

}  // namespace hashemb
