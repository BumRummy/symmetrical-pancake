import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from Crypto.PublicKey import RSA
from Crypto.Util.number import bytes_to_long, long_to_bytes

# Define the range for RSA key generation
min_key = 20000000000000000
max_key = 0x3ffffffffffffffff

# Query the user for the keyspace size
keyspace_size = query_keyspace()

# Define the target public key in hex format
target_public_key_hex = "13zb1hQbWVsc2S7ZTZnP2G4undNNpdh5so"

# Convert the hex public key to bytes
target_public_key_bytes = bytes.fromhex(target_public_key_hex)

# Convert the bytes to a long integer
target_public_key_int = bytes_to_long(target_public_key_bytes)

# Define the RSA key length in bits
key_length = 2048  # RSA key length in bits

# Kernel code for generating RSA key pairs
kernel_code = """
#include <curand_kernel.h>
#include <stdio.h>

#define RSA_BITS 2048

extern "C" {
__global__ void generate_rsa_key_pairs(unsigned long long seed, unsigned char* public_keys, unsigned long long target_public_key, int keyspace_size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= keyspace_size) return;
    curandState_t state;
    curand_init(seed, tid, 0, &state);
    RSA* rsa;
    unsigned char* public_key_buffer = new unsigned char[RSA_BITS/8];
    while (1) {
        rsa = RSA_generate_key(RSA_BITS, 65537, nullptr, nullptr);
        RSA_public_key_bytes(rsa, public_key_buffer);
        unsigned long long public_key = *((unsigned long long*)public_key_buffer);
        if (public_key == target_public_key) {
            printf("Target public key found!\\n");
            delete[] public_key_buffer;
            break;
        }
        if (tid == 0 && (public_key % 10000) == 0) {
            printf("Generated public key: %%llx\\n", public_key);
        }
        delete rsa;
    }
    public_keys[tid * (RSA_BITS/8)] = public_key_buffer[0];
    public_keys[tid * (RSA_BITS/8) + 1] = public_key_buffer[1];
    public_keys[tid * (RSA_BITS/8) + 2] = public_key_buffer[2];
    delete[] public_key_buffer;
}
}
"""

# Compile the CUDA kernel code
module = drv.SourceModule(kernel_code)
generate_rsa_key_pairs = module.get_function("generate_rsa_key_pairs")

# Allocate memory for storing the generated public keys
public_keys = np.zeros((keyspace_size, key_length // 8), dtype=np.uint8)

# Define the block and grid dimensions
block_size = 256
grid_size = (keyspace_size + block_size - 1) // block_size

# Generate RSA key pairs using CUDA
generate_rsa_key_pairs(np.uint64(123456), drv.Out(public_keys), np.uint64(target_public_key_int), np.int32(keyspace_size), block=(block_size, 1, 1), grid=(grid_size, 1))

# Convert the generated public keys to hexadecimal format
hex_public_keys = [public_key.tobytes().hex() for public_key in public_keys]

# Print the generated public keys
for i, hex_key in enumerate(hex_public_keys):
    print("Target public key {} found after {} keys.".format(current_key_hex, num_keys_generated))

# Save the generated public keys to a file
with open("generated_public_keys.txt", "w") as f:
    for hex_key in hex_public_keys:
        f.write(f"{hex_key}\n")



