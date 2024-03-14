import cupy as cp
import time
import hashlib
import math

def generate_prime():
    n = cp.random.randint(2**15, 2**16)
    while not cp.all(cp.isprime(n)):
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

if __name__ == "__main__":
    main()
