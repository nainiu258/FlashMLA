#include "../phase1.h"
#include "../phase1.cuh"

namespace sm90::fwd {

template void run_fwd_phase1_kernel_dsmem_crossover<576, false, true>(const SparseAttnFwdParams& params);

}  // namespace sm90::fwd
