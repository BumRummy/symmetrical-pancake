import cupy as cp
import time
import hashlib
import math

def generate_primes(n):
    sieve = cp.ones(n, dtype=bool)
    sieve[:2] = False
    for i in range(2, int(math.sqrt(n)) + 1):
        if sieve[i]:
            sieve[i*i:n:i] = False
    primes = cp.where(sieve)[0]
    return primes

def generate_keypair(primes):
    p = cp.random.choice(primes, size=(1,))
    q = cp.random.choice(primes, size=(1,))
    while q == p:
        q = cp.random.choice(primes, size=(1,))

    n = p * q
    phi = (p - 1) * (q - 1)

    e = 65537  # Commonly used value for e

    d = mod_inverse(e, phi)

    return ((e, n), (d, n))

def mod_inverse(a, m):
    m0, x0, x1 = m, 0, 1
    while a > 1:
        q = a // m
        m, a = a % m, m
        x0, x1 = x1 - q * x0, x0
    return int(x1 + m0) if x1 < 0 else int(x1)

def hash_public_key(key):
    return hashlib.sha256(str(key).encode()).hexdigest()

def main():
    target_public_key = "13zb1hQbWVsc2S7ZTZnP2G4undNNpdh5so"
    target_hash = hashlib.sha256(target_public_key.encode()).hexdigest()

    primes = generate_primes(10**6)

    start_time = time.time()
    attempts = 0
    while True:
        attempts += 1
        public_key, _ = generate_keypair(primes)
        hashed_public_key = hash_public_key(public_key)
        if hashed_public_key == target_hash:
            print("\nRSA Key Pair Found:")
            print("Public Key:", public_key)
            print("Attempts:", attempts)
            break

        if attempts % 100000 == 0:
            elapsed_time = time.time() - start_time
            speed = (attempts / elapsed_time) / 1e6
            print(f"\rSpeed: {speed:.2f} mkey/s", end="", flush=True)

if __name__ == "__main__":
    main()
