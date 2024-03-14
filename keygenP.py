import cupy as cp
import time
import hashlib

def generate_keypair():
    # Generate random primes within the given range
    p = cp.random.randint(20000000, 30000000)
    q = cp.random.randint(20000000, 30000000)

    # Ensure p and q are distinct primes
    while not is_prime(p):
        p = cp.random.randint(20000000, 30000000)
    while not is_prime(q) or q == p:
        q = cp.random.randint(20000000, 30000000)

    # Calculate n and phi(n)
    n = p * q
    phi = (p - 1) * (q - 1)

    # Choose e such that e is coprime with phi(n)
    e = cp.random.randint(2, phi - 1)
    while gcd(e, phi) != 1:
        e = cp.random.randint(2, phi - 1)

    # Compute the modular multiplicative inverse of e mod phi(n)
    d = mod_inverse(e, phi)

    return ((e, n), (d, n))

def is_prime(num):
    if num <= 1:
        return False
    if num <= 3:
        return True
    if num % 2 == 0 or num % 3 == 0:
        return False
    i = 5
    while i * i <= num:
        if num % i == 0 or num % (i + 2) == 0:
            return False
        i += 6
    return True

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def mod_inverse(a, m):
    m0, x0, x1 = m, 0, 1
    while a > 1:
        q = a // m
        m, a = a % m, m
        x0, x1 = x1 - q * x0, x0
    return x1 + m0 if x1 < 0 else x1

def hash_public_key(key):
    return hashlib.sha256(str(key).encode()).hexdigest()

def main():
    target_public_key = "13zb1hQbWVsc2S7ZTZnP2G4undNNpdh5so"

    start_time = time.time()
    last_print_time = start_time
    attempts = 0
    while True:
        attempts += 1
        public_key, _ = generate_keypair()
        hashed_public_key = hash_public_key(public_key)
        if hashed_public_key == target_public_key:
            print("\nRSA Key Pair Found:")
            print("Public Key:", public_key)
            print("Attempts:", attempts)
            break
        if attempts % 100000 == 0:
            elapsed_time = time.time() - start_time
            completion_percentage = (attempts / 100000) * 100
            speed = (attempts / elapsed_time) / 1e6
            print(f"Speed: {speed:.2f} mkey/s", end="\r", flush=True)

if __name__ == "__main__":
    main()
