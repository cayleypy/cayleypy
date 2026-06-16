#include "random_walks.h"

#include <torch/extension.h>
#include <cstdint>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

using torch::Tensor;
using torch::TensorOptions;

WalksCpp::WalksCpp(int num_walks, int walks_len, int state_size, TensorOptions options) {
    states = torch::empty({num_walks, walks_len, state_size}, options.dtype(torch::kLong));
    distances = torch::empty({num_walks, walks_len},            options.dtype(torch::kLong));
}

WalksCpp random_walks_classic_cpp(
    Tensor gens,            
    Tensor central_state,  
    const int64_t num_walks,
    const int64_t walks_len,
    int num_omp_threads
){
    TORCH_CHECK(gens.device().is_cpu(), "CPU only implementation (for now).");
    TORCH_CHECK(central_state.device().is_cpu(), "CPU only implementation (for now).");
    TORCH_CHECK(gens.dtype() == torch::kLong, "gens must be int64 (LongTensor).");
    TORCH_CHECK(central_state.dtype() == torch::kLong, "central_state must be int64 (LongTensor).");
    TORCH_CHECK(gens.dim() == 2, "gens must be 2D [num_gens, state_size].");
    TORCH_CHECK(central_state.dim() == 1, "central_state must be 1D [state_size].");
    TORCH_CHECK(num_walks >= 0 && walks_len >= 0, "num_walks and walks_len must be non-negative.");

    const int64_t num_gens   = gens.size(0);
    const int64_t state_size = gens.size(1);

    gens = gens.contiguous();
    central_state = central_state.contiguous();

    TensorOptions tensor_options = central_state.options();
    WalksCpp walks(num_walks, walks_len, state_size, tensor_options);

    // first state is central_state, so only need (walks_len - 1) random choices
    Tensor choices = torch::randint(num_gens, {num_walks, walks_len-1}, tensor_options.dtype(torch::kLong)); 

    const int64_t* gens_ptr = gens.data_ptr<int64_t>();
    const int64_t* central_state_ptr = central_state.data_ptr<int64_t>();
    const int64_t* choices_ptr = choices.data_ptr<int64_t>();
    int64_t* states_ptr = walks.states.data_ptr<int64_t>();
    int64_t* distances_ptr = walks.distances.data_ptr<int64_t>();
    int64_t choices_row_len = walks_len - 1;

    #ifdef _OPENMP
        if (num_omp_threads > 0) {
            omp_set_num_threads(num_omp_threads);
        }
    #endif

    #pragma omp parallel for if(num_omp_threads) schedule(static)
    for (int64_t iw = 0; iw < num_walks; ++iw) {
        const int64_t* current_ptr = central_state_ptr;  
        const int64_t offset = iw * walks_len;

        // first state  is just the central state, all distances are 0
        int64_t* out_ptr0 = states_ptr + (iw * walks_len * state_size);
        std::memcpy(out_ptr0, current_ptr, static_cast<size_t>(state_size) * sizeof(int64_t));
        std::memset(distances_ptr + offset, 0, sizeof(int64_t));

        for (int64_t js = 1; js < walks_len; ++js) {
            // choices start from step 1, as step 0 is central_state, so js - 1
            const int64_t choice = choices_ptr[iw * choices_row_len + js - 1]; 
            const int64_t* gen_row = gens_ptr + choice * state_size;
            int64_t* out_ptr = states_ptr + ((offset + js) * state_size);
            // apply generator
            for (int64_t k = 0; k < state_size; ++k) {
                out_ptr[k] = current_ptr[gen_row[k]];
            }
            current_ptr = out_ptr;

            // distances are estimated as numbers of random walk steps
            distances_ptr[offset + js] = js;
        }
    }

    return walks;
}

