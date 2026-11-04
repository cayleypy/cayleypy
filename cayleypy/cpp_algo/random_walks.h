#pragma once

#include <torch/extension.h>

using torch::Tensor;

struct WalksCpp {
    Tensor states;
    Tensor distances;

    WalksCpp() = default; 
    WalksCpp(int num_walks, int walks_len, int state_size, torch::TensorOptions options);
};

WalksCpp random_walks_classic_cpp(
    Tensor gens, 
    Tensor central_state, 
    const int64_t num_walks, 
    const int64_t walks_len,
    int num_omp_threads
);