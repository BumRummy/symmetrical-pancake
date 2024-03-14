import numpy as np
from numba import cuda

# Define the CUDA kernel code
cuda_kernel_code = """
#include <curand_kernel.h>
#include <stdint.h>

extern "C" {
__global__ void generate_rsa_key_pairs(uint64_t* output, uint64_t min_key, uint64_t max_key) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(clock64() + idx, 0, 0, &state); // Initialize random number generator

    while (1) {
        uint64_t n = curand(&state) % (max_key - min_key) + min_key;
        uint64_t e = curand(&state) % (max_key - min_key) + min_key;
        uint64_t d = curand(&state) % (max_key - min_key) + min_key;

        output[3 * idx] = n;
        output[3 * idx + 1] = e;
        output[3 * idx + 2] = d;

        if (output[3 * idx] == 0x13zb1hQbWVsc2S7ZTZnP2G4undNNpdh5soULL) // Check if public key matches
            break;
    }
}
}
"""

# Compile the CUDA kernel code
cuda_kernel = cuda.jit('void(uint64[:], uint64, uint64)', device=True)(cuda_kernel_code)

# Define the parameters for key space
min_key = np.uint64(20000000000000000)
max_key = np.uint64(0x3ffffffffffffffff)

# Allocate memory on the host for the output
output = np.zeros((1, 3), dtype=np.uint64)

# Allocate memory on the device for the output
d_output = cuda.to_device(output)

# Launch the CUDA kernel
threads_per_block = 256
blocks_per_grid = 1
cuda_kernel[blocks_per_grid, threads_per_block](d_output, min_key, max_key)

# Copy the result back from the device to the host
output = d_output.copy_to_host()

# Print the generated RSA key pairs
print("Generated RSA key pairs:")
print("N:", hex(output[0][0]))
print("E:", hex(output[0][1]))
print("D:", hex(output[0][2]))
