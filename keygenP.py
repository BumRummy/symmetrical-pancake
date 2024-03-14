import cupy as cp
import time
import hashlib
import math

def is_prime(n, k=5):
    """Miller-Rabin primality test."""
    if n <= 3:
        return n == 2 or n == 3
    if n % 2 == 0:
        return False

    def check(a, s, d, n):
        x = pow(a, d, n)
        if x == 1:
            return True
        for i in range(s - 1):
            if x == n - 1:
                return True
            x = pow(x, 2, n)
        return x == n - 1

    s = 0
    d = n - 1
    while d % 2 == 0:
        d //= 2
        s += 1

    for _ in range(k):
        a = cp.random.randint(2, n - 1)
        if not check(a, s, d, n):
            return False

    return True

def generate_prime():
    n = cp.random.randint(2**15, 2**16)
    while not is_prime(n):
        n = cp.random.randint(2**15, 2**16)
    return int(n)

def generate_keypair():
    p = generate_prime()
    q = generate_prime()

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

        if attempts % 10 == 0:
            elapsed_time = time.time() - start_time
            speed = attempts / elapsed_time
            print(f"Speed: {speed:.2f} hashes/s", end="\r", flush=True)

if __name__ == "__main__":
    main()
