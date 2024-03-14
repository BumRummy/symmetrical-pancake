import cupy as cp
import time
import hashlib

def generate_keypair(primes):
    p_index = cp.random.randint(len(primes))
    q_index = cp.random.randint(len(primes))
    while q_index == p_index:
        q_index = cp.random.randint(len(primes))

    p = primes[p_index]
    q = primes[q_index]

    n = p * q
    phi = (p - 1) * (q - 1)

    e = cp.random.randint(2, phi - 1)
    while cp.gcd(e, phi) != 1:
        e = cp.random.randint(2, phi - 1)

    d = mod_inverse(e, phi)

    return ((e, n), (d, n))

def generate_keypair(primes):
    p = cp.random.choice(primes)
    q = cp.random.choice(primes)
    while q == p:
        q = cp.random.choice(primes)

    n = p * q
    phi = (p - 1) * (q - 1)

    e = cp.random.randint(2, phi - 1)
    while cp.gcd(e, phi) != 1:
        e = cp.random.randint(2, phi - 1)

    d = mod_inverse(e, phi)

    return ((e, n), (d, n))

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
    target_hash = hashlib.sha256(target_public_key.encode()).hexdigest()

    print("Generating prime numbers...")
    primes = generate_primes()
    print("Prime numbers generated.")

    start_time = time.time()
    last_print_time = start_time
    attempts = 0
    while True:
        attempts += 1
        print(f"\nAttempt {attempts}:")
        public_key, _ = generate_keypair(primes)
        hashed_public_key = hash_public_key(public_key)
        if hashed_public_key == target_hash:
            print("\nRSA Key Pair Found:")
            print("Public Key:", public_key)
            print("Attempts:", attempts)
            break
        if attempts % 100000 == 0:
            elapsed_time = time.time() - start_time
            completion_percentage = (attempts / 100000) * 100
            speed = (attempts / elapsed_time) / 1e6
            print(f"Speed: {speed:.2f} mkey/s", end="\r", flush=True)
        else:
            speed = (attempts / (time.time() - start_time)) / 1e6
            print(f"Speed: {speed:.2f} mkey/s", end="\r", flush=True)

if __name__ == "__main__":
    main()
