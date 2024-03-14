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

    d = pow(e, -1, phi)  # Computing modular inverse directly

    return ((e, n), (d, n))

def hash_public_key(key):
    return hashlib.sha256(str(key).encode()).hexdigest()

def main():
    target_public_key = "13zb1hQbWVsc2S7ZTZnP2G4undNNpdh5so"
    target_hash = hashlib.sha256(target_public_key.encode()).hexdigest()

    print("Generating prime numbers...")
    primes = generate_primes(10**6)
    print("Prime numbers generated.")

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
            print(f"Speed: {speed:.2f} mkey/s", end="\r", flush=True)

if __name__ == "__main__":
    main()
