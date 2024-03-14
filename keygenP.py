import numpy as np
from numba import cuda
import hashlib

# CUDA kernel to generate RSA key pairs
@cuda.jit
def generate_rsa_key_pairs(output, min_key, max_key, target_public_key):
    idx = cuda.grid(1)
    while True:
        n = idx + min_key
        e = idx + min_key + 1
        d = idx + min_key + 2

        # Check if the generated public key matches the target
        public_key = hashlib.sha256(f"{n},{e}".encode()).hexdigest()
        if public_key == target_public_key:
            output[0] = n
            output[1] = e
            output[2] = d
            break

# Define the keyspace range
min_key = np.uint64(0x2000000000000000)  # 20000000000000000
max_key = np.uint64(0x3fffffffffffffff)  # 3ffffffffffffffff
target_public_key = "13zb1hQbWVsc2S7ZTZnP2G4undNNpdh5so"  # Target public key

# Initialize output array to store the RSA key pairs
output = np.zeros(3, dtype=np.uint64)

# Set up CUDA grid and block dimensions
block_dim = 256
grid_dim = (max_key - min_key) // block_dim + 1

# Launch the CUDA kernel
generate_rsa_key_pairs[grid_dim, block_dim](output, min_key, max_key, target_public_key)

# Print the generated RSA key pairs
print("Generated RSA key pairs:")
print("N:", hex(output[0]))
print("E:", hex(output[1]))
print("D:", hex(output[2]))
