#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>
#include "embedding_table.h"

namespace py = pybind11;
namespace hashemb {

// ===========================================================================
// OptimizerConfig conversion helpers
// ===========================================================================

inline OptimizerConfig config_from_kwargs(const std::string& opt,
                                           float lr, float beta1,
                                           float beta2, float eps) {
  OptimizerConfig cfg;
  cfg.lr = lr;
  if (opt == "adam") {
    cfg.type = OptimizerConfig::ADAM;
    cfg.beta1 = beta1;
    cfg.beta2 = beta2;
    cfg.eps = eps;
  }
  return cfg;
}

// ===========================================================================
// Python wrapper
// ===========================================================================

class PyHashEmbedding {
 public:
  PyHashEmbedding(int64_t initial_capacity, int32_t embedding_dim,
                  const std::string& opt = "sgd",
                  float lr = 0.01f,
                  float beta1 = 0.9f, float beta2 = 0.999f,
                  float eps = 1e-8f, int64_t block_size = 10'000'000)
      : table_(initial_capacity, embedding_dim,
               config_from_kwargs(opt, lr, beta1, beta2, eps),
               block_size) {}

  // ── Lookup ──────────────────────────────────────────────────────────

  py::tuple lookup_and_gather(py::array_t<int64_t> keys) {
    py::buffer_info buf = keys.request();
    int64_t n = buf.shape[0];
    auto* keys_ptr = static_cast<const int64_t*>(buf.ptr);

    auto embeddings = make_float_array(n, table_.embedding_dim());
    auto slot_indices = py::array_t<int32_t>(static_cast<ssize_t>(n));

    auto* emb_ptr = static_cast<float*>(embeddings.request().ptr);
    auto* slot_ptr = static_cast<int32_t*>(slot_indices.request().ptr);

    table_.lookup_and_gather(keys_ptr, emb_ptr, slot_ptr, n);
    return py::make_tuple(embeddings, slot_indices);
  }

  py::array_t<int32_t> find_or_create(py::array_t<int64_t> keys) {
    py::buffer_info buf = keys.request();
    int64_t n = buf.shape[0];
    auto* keys_ptr = static_cast<const int64_t*>(buf.ptr);

    auto slot_indices = py::array_t<int32_t>(static_cast<ssize_t>(n));
    auto* slot_ptr = static_cast<int32_t*>(slot_indices.request().ptr);

    table_.find_or_create(keys_ptr, slot_ptr, n);
    return slot_indices;
  }

  py::array_t<float> lookup(py::array_t<int32_t> slot_indices) {
    py::buffer_info buf = slot_indices.request();
    int64_t n = buf.shape[0];
    auto* slot_ptr = static_cast<const int32_t*>(buf.ptr);

    auto embeddings = make_float_array(n, table_.embedding_dim());
    auto* emb_ptr = static_cast<float*>(embeddings.request().ptr);

    table_.lookup(slot_ptr, emb_ptr, n);
    return embeddings;
  }

  // ── Gradient / Optimiser ────────────────────────────────────────────

  void scatter_add_grad(py::array_t<int32_t> slot_indices,
                        py::array_t<float> grads) {
    py::buffer_info si_buf = slot_indices.request();
    py::buffer_info g_buf = grads.request();
    int64_t n = si_buf.shape[0];
    table_.scatter_add_grad(
        static_cast<const int32_t*>(si_buf.ptr),
        static_cast<const float*>(g_buf.ptr), n);
  }

  void step() { table_.step(); }
  void zero_grad() { table_.zero_grad(); }

  // ── Serialisation ───────────────────────────────────────────────────

  py::dict state_dict() {
    std::vector<int64_t> keys;
    std::vector<int32_t> slots;
    std::vector<float> weight, grad, m, v;
    int64_t t = 0;
    std::string opt_type;

    table_.state_dict_arrays(keys, slots, weight, grad, m, v, t, opt_type);

    int64_t n = static_cast<int64_t>(keys.size());
    int32_t D = table_.embedding_dim();

    auto np_keys   = py::array_t<int64_t>(static_cast<ssize_t>(n));
    auto np_slots  = py::array_t<int32_t>(static_cast<ssize_t>(n));
    auto np_weight = make_float_array(n, D);
    auto np_grad   = make_float_array(n, D);
    auto np_m      = make_float_array(n, D);
    auto np_v      = make_float_array(n, D);

    std::memcpy(np_keys.request().ptr,   keys.data(),   sizeof(int64_t) * n);
    std::memcpy(np_slots.request().ptr,  slots.data(),  sizeof(int32_t) * n);
    std::memcpy(np_weight.request().ptr, weight.data(), sizeof(float) * n * D);
    std::memcpy(np_grad.request().ptr,   grad.data(),   sizeof(float) * n * D);
    std::memcpy(np_m.request().ptr,      m.data(),      sizeof(float) * n * D);
    std::memcpy(np_v.request().ptr,      v.data(),      sizeof(float) * n * D);

    py::dict d;
    d["keys"]       = np_keys;
    d["slots"]      = np_slots;
    d["weight"]     = np_weight;
    d["grad"]       = np_grad;
    d["m"]          = np_m;
    d["v"]          = np_v;
    d["t"]          = py::int_(t);
    d["opt_type"]   = py::str(opt_type);
    d["dim"]        = py::int_(table_.embedding_dim());
    return d;
  }

  void load_state_dict(py::dict d) {
    auto np_keys   = d["keys"].cast<py::array_t<int64_t>>();
    auto np_slots  = d["slots"].cast<py::array_t<int32_t>>();
    auto np_weight = d["weight"].cast<py::array_t<float>>();
    auto np_grad   = d["grad"].cast<py::array_t<float>>();
    auto np_m      = d["m"].cast<py::array_t<float>>();
    auto np_v      = d["v"].cast<py::array_t<float>>();
    int64_t t      = d["t"].cast<int64_t>();
    std::string opt_type = d["opt_type"].cast<std::string>();
    int64_t n = np_keys.request().size;

    table_.load_state_dict_arrays(
        n,
        static_cast<const int64_t*>(np_keys.request().ptr),
        static_cast<const int32_t*>(np_slots.request().ptr),
        static_cast<const float*>(np_weight.request().ptr),
        static_cast<const float*>(np_grad.request().ptr),
        static_cast<const float*>(np_m.request().ptr),
        static_cast<const float*>(np_v.request().ptr),
        t, opt_type);
  }

  // ── Accessors ───────────────────────────────────────────────────────

  int64_t capacity() const { return table_.initial_capacity(); }
  int32_t embedding_dim() const { return table_.embedding_dim(); }
  int64_t num_entries() const { return table_.num_entries(); }

 private:
  EmbeddingTable table_;

  static py::array_t<float> make_float_array(int64_t n, int32_t d) {
    std::vector<ssize_t> shape = {static_cast<ssize_t>(n), static_cast<ssize_t>(d)};
    return py::array_t<float>(shape);
  }
};

}  // namespace hashemb

PYBIND11_MODULE(_hashemb_cpp, m) {
  m.doc() = "HashEmb C++ extension";

  py::class_<hashemb::PyHashEmbedding>(m, "HashEmbeddingTable")
      .def(py::init<int64_t, int32_t, std::string, float, float, float, float, int64_t>(),
           py::arg("capacity"), py::arg("embedding_dim"),
           py::arg("optimizer") = "sgd",
           py::arg("lr") = 0.01f,
           py::arg("beta1") = 0.9f,
           py::arg("beta2") = 0.999f,
           py::arg("eps") = 1e-8f,
           py::arg("block_size") = 10'000'000)

      // Lookup
      .def("lookup_and_gather", &hashemb::PyHashEmbedding::lookup_and_gather,
           py::arg("keys"))
      .def("find_or_create", &hashemb::PyHashEmbedding::find_or_create,
           py::arg("keys"))
      .def("lookup", &hashemb::PyHashEmbedding::lookup,
           py::arg("slot_indices"))

      // Optimizer
      .def("scatter_add_grad", &hashemb::PyHashEmbedding::scatter_add_grad,
           py::arg("slot_indices"), py::arg("grads"))
      .def("step", &hashemb::PyHashEmbedding::step)
      .def("zero_grad", &hashemb::PyHashEmbedding::zero_grad)

      // Serialization
      .def("state_dict", &hashemb::PyHashEmbedding::state_dict)
      .def("load_state_dict", &hashemb::PyHashEmbedding::load_state_dict,
           py::arg("state_dict"))

      // Properties
      .def_property_readonly("capacity", &hashemb::PyHashEmbedding::capacity)
      .def_property_readonly("embedding_dim", &hashemb::PyHashEmbedding::embedding_dim)
      .def_property_readonly("num_entries", &hashemb::PyHashEmbedding::num_entries);
}
