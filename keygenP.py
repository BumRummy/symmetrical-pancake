import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
import rsa
import time

# Define the CUDA kernel code for generating RSA key pairs
kernel_code = """
#include <curand_kernel.h>
extern "C" {
    __global__ void generate_rsa_key_pairs(char* public_keys, char* private_keys, unsigned int* counter) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        // Initialize a random number generator
        curandState state;
        curand_init(idx, 0, 0, &state);

        // Generate a random RSA key pair
        rsa.generate_keypair(public_keys + idx * 128, private_keys + idx * 128, &state);

        // Increment the counter
        atomicAdd(counter, 1);
    }
}
"""

# Compile the CUDA kernel code
module = SourceModule(kernel_code)

# Get the CUDA function
generate_rsa_key_pairs = module.get_function("generate_rsa_key_pairs")

# Define the keyspace
keyspace = np.array([0x2000000000000000, 0x3ffffffffffffffff], dtype=np.uint64)

# Define the number of keys to generate per GPU
num_keys_per_gpu = 1000

# Define the interval for printing speed in seconds
print_interval = 30

# Allocate memory on the device for keys and counter
public_keys_gpu = cuda.mem_alloc(128 * num_keys_per_gpu)
private_keys_gpu = cuda.mem_alloc(128 * num_keys_per_gpu)
counter_gpu = cuda.mem_alloc(np.dtype(np.uint32).itemsize)

# Set up grid and block dimensions
block_size = 256
grid_size = (num_keys_per_gpu + block_size - 1) // block_size

# Start time
start_time = time.time()

try:
    while True:
        # Reset the counter
        cuda.memcpy_htod(counter_gpu, np.array([0], dtype=np.uint32))

        # Generate RSA key pairs on each GPU
        generate_rsa_key_pairs(public_keys_gpu, private_keys_gpu, counter_gpu, block=(block_size, 1, 1), grid=(grid_size, 1))

        # Copy the counter back to the host
        counter = np.zeros(1, dtype=np.uint32)
        cuda.memcpy_dtoh(counter, counter_gpu)

        # Calculate elapsed time
        elapsed_time = time.time() - start_time

        # Calculate speed in mkeys per second
        speed_mkeys_per_sec = (counter[0] / 1000000) / elapsed_time

        # Print speed per GPU
        print("Speed per GPU: {:.2f} mkeys/s".format(speed_mkeys_per_sec))

        # Reset start time
        start_time = time.time()

        # Wait for the print interval
        time.sleep(print_interval)

except KeyboardInterrupt:
    print("Process interrupted")
