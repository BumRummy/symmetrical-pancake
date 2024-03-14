import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import rsa  # For RSA key generation
import time

# CUDA kernel code for RSA key generation
cuda_kernel_code = """
#include <curand_kernel.h>
#include <stdio.h>
#include <time.h>

extern "C" {
__global__ void generate_rsa_key_pairs(char* public_keys, char* private_keys, int keyspace_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize random seed
    curandState state;
    curand_init(clock64(), idx, 0, &state);

    // Generate RSA key pairs
    if (idx < keyspace_size) {
        rsa_key_pair pair = rsa_new_key_pair(2048, 65537);
        rsa_key_to_string(pair.public_key, public_keys + idx * PUBLIC_KEY_SIZE);
        rsa_key_to_string(pair.private_key, private_keys + idx * PRIVATE_KEY_SIZE);
    }
}
}
"""

# Function to compile CUDA kernel
def compile_cuda_kernel():
    mod = SourceModule(cuda_kernel_code)
    return mod.get_function("generate_rsa_key_pairs")

# Function to generate RSA key pairs on GPU
def generate_rsa_key_pairs_on_gpu(keyspace_size):
    # Define parameters
    block_size = 256
    grid_size = (keyspace_size + block_size - 1) // block_size

    # Allocate memory on the GPU
    public_keys_gpu = cuda.mem_alloc(keyspace_size * rsa.PUBLIC_KEY_SIZE)
    private_keys_gpu = cuda.mem_alloc(keyspace_size * rsa.PRIVATE_KEY_SIZE)

    # Launch the CUDA kernel
    generate_rsa_key_pairs(public_keys_gpu, private_keys_gpu, np.int32(keyspace_size), block=(block_size, 1, 1), grid=(grid_size, 1))

    # Copy the results back to the host
    public_keys = np.empty((keyspace_size, rsa.PUBLIC_KEY_SIZE), dtype=np.uint8)
    private_keys = np.empty((keyspace_size, rsa.PRIVATE_KEY_SIZE), dtype=np.uint8)
    cuda.memcpy_dtoh(public_keys, public_keys_gpu)
    cuda.memcpy_dtoh(private_keys, private_keys_gpu)

    return public_keys

# Function to calculate speed in Mkeys
def calculate_speed(num_keys_generated, start_time, end_time):
    elapsed_time = end_time - start_time
    speed = num_keys_generated / elapsed_time
    speed_in_Mkeys = speed / 1e6
    return speed_in_Mkeys

# Function to generate RSA key pairs within the specified keyspace
def generate_rsa_key_pairs_in_keyspace(start_key, end_key):
    start_key_int = int(start_key, 16)
    end_key_int = int(end_key, 16)
    keyspace_size = end_key_int - start_key_int + 1

    # Compile CUDA kernel
    generate_rsa_key_pairs = compile_cuda_kernel()

    # Initialize variables for speed calculation
    start_time = time.time()
    num_keys_generated = 0

    # Main loop for key generation
    while True:
        public_keys = generate_rsa_key_pairs_on_gpu(keyspace_size)
        num_keys_generated += keyspace_size

        # Check if any of the generated keys fall within the specified keyspace
        for idx in range(keyspace_size):
            current_key_int = start_key_int + idx
            current_key_hex = hex(current_key_int)[2:].zfill(16)
            current_key_bytes = bytes.fromhex(current_key_hex)
            if current_key_bytes in public_keys.tobytes():
                end_time = time.time()
                speed_in_Mkeys = calculate_speed(num_keys_generated, start_time, end_time)
                print(f"Target public key {current_key_hex} found after {num_keys_generated} keys.")
                print(f"Speed: {speed_in_Mkeys:.2f} Mkeys/s")
                return

# Define the keyspace
start_key = '0000000000000000'
end_key

