#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "random_walks.h"

namespace py = pybind11;

PYBIND11_MODULE(cpp_algo, m) {
    m.doc() = "pybind11 bindings for CayleyPy C++ algorithms";
    py::class_<WalksCpp>(m, "WalksCpp")
    .def_property_readonly("states",    [](const WalksCpp& w){ return w.states; })
    .def_property_readonly("distances", [](const WalksCpp& w){ return w.distances; });

    m.def("random_walks_classic_cpp",
          &random_walks_classic_cpp,
          R"pbdoc(
          Vanilla C++ random walks function with optional OpenMP parallelization
          
          Args:
              gens (LongTensor): Generators tensor of shape [num_gens, state_size].
              central_state (LongTensor): Central state tensor of shape [state_size].
              num_walks (int): Number of random walks to perform.
              walks_len (int): Length of each random walk.
              num_omp_threads (int, optional): Number of OpenMP threads to use. Setting this to 0 disables parallelization. Default is 0.
          Returns:
              LongTensor: Tensor of shape [num_walks, walks_len, state_size] containing the random walks.
          )pbdoc",
          py::arg("gens"),
          py::arg("central_state"),
          py::arg("num_walks"),
          py::arg("walks_len"),
          py::arg("num_omp_threads") = 0,
          py::return_value_policy::move  
      );
}
