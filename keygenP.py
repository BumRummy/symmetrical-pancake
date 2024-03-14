import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import time
import hashlib
from pycuda.compiler import SourceModule

# CUDA kernel for Miller-Rabin primality test
miller_rabin_kernel = """
__device__ bool is_prime(unsigned long long int n) {
    if (n <= 3) return (n == 2 || n == 3);
    if (n % 2 == 0) return false;
    
    unsigned long long int d = n - 1;
    int s = 0;
    while (d % 2 == 0) {
        d >>= 1;
        s++;
    }
    
    unsigned long long int bases[5] = {2, 3, 5, 7, 11};
    for (int i = 0; i < 5; i++) {
        unsigned long long int a = bases[i];
        unsigned long long int x = 1;
        for (unsigned long long int exp = d; exp > 0; exp >>= 1) {
            if (exp & 1) x = (x * a) % n;
            a = (a * a) % n;
        }
        if (x == 1 || x == n - 1) continue;
        for (int r = 1; r < s; r++) {
            x = (x * x) % n;
            if (x == 1) return false;
            if (x == n - 1) break;
        }
        if (x != n - 1) return false;
    }
    return true;
}
"""

# Compile CUDA kernel
mod = SourceModule(miller_rabin_kernel)

# Get CUDA function
is_prime_cuda = mod.get_function("is_prime")

# Generate prime numbers using CUDA
def generate_prime():
    n = np.random.randint(2**50, 2**61)
    n |= 1  # Make sure it's odd
    while not is_prime_cuda(np.uint64(n), block=(1,1,1), grid=(1,1,1))[0]:
        n += 2
    return n

# Compute modular inverse using CUDA
def mod_inverse(a, m):
    a = np.uint64(a)
    m = np.uint64(m)
    x0, x1 = np.uint64(0), np.uint64(1)
    while a > 1:
        q = a // m
        m, a = a % m, m
        x0, x1 = x1 - q * x0, x0
    return (x1 + np.uint64(m)) if x1 < 0 else x1

# Generate RSA key pair
def generate_keypair():
    p = generate_prime()
    q = generate_prime()

    n = p * q
    phi = (p - 1) * (q - 1)

    e = 65537  # Commonly used value for e
    d = mod_inverse(e, phi)  # Computing modular inverse using CUDA

    return ((e, n), (d, n))

# Hash public key
def hash_public_key(key):
    return hashlib.sha256(str(key).encode()).hexdigest()

# Main function
def main():
    target_public_key = "13zb1hQbWVsc2S7ZTZnP2G4undNNpdh5so"
    target_hash = hashlib.sha256(target_public_key.encode()).hexdigest()

    print("Searching for the target public key...")

    start_time = time.time()
    attempts = 0
    while True:
        attempts += 1
        public_key, private_key = generate_keypair()
        hashed_public_key = hash_public_key(public_key)
        if hashed_public_key == target_hash:
            print("\nRSA Key Pair Found:")
            print("Public Key:", public_key)
            print("Private Key:", private_key)
            print("Attempts:", attempts)
            break

if __name__ == "__main__":
    main()
