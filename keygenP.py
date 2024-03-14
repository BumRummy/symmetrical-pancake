import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import time

# Define the kernel code for generating RSA key pairs
kernel_code = """
#include <curand_kernel.h>

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
"""

# Compile the kernel code
mod = SourceModule(kernel_code)

# Get the CUDA function
generate_rsa_key_pairs = mod.get_function("generate_rsa_key_pairs")

# Define the keyspace
min_key = 0x2000000000000000
max_key = 0x3ffffffffffffffff

# Define the number of threads per block and number of blocks
block_size = 256
grid_size = 30

# Define the size of output array
output_size = grid_size * block_size * 3
output = np.zeros(output_size, dtype=np.uint64)

# Execute the kernel
start_time = time.time()
generate_rsa_key_pairs(cuda.Out(output), np.uint64(min_key), np.uint64(max_key), block=(block_size, 1, 1), grid=(grid_size, 1))
cuda.Context.synchronize()
end_time = time.time()

# Calculate the speed
speed = output_size / (end_time - start_time) / 10**6  # Convert to Mkey/s

print("Speed per GPU:", speed, "Mkey/s")
