import numpy as np
import time
import hashlib

# Custom modular exponentiation function
def mod_exp(base, exponent, modulus):
    result = 1
    while exponent > 0:
        if exponent % 2 == 1:
            result = (result * base) % modulus
        exponent //= 2
        base = (base * base) % modulus
    return result

# Miller-Rabin primality test
def is_prime(n, num_trials=5):
    if n <= 3:
        return n == 2 or n == 3
    if n % 2 == 0:
        return False
    
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    
    bases = [2, 3, 5, 7, 11]
    for a in bases[:num_trials]:
        x = mod_exp(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = mod_exp(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True

# Generate prime number of given bit length
def generate_prime(bit_length):
    while True:
        n = np.random.randint(0, 2**bit_length)
        n |= 1  # Make sure it's odd
        if is_prime(n):
            return n

# Generate RSA key pair
def generate_keypair(bit_length):
    p = generate_prime(bit_length)
    q = generate_prime(bit_length)

    n = p * q
    phi = (p - 1) * (q - 1)

    e = 65537  # Commonly used value for e
    d = mod_inverse(e, phi)  # Computing modular inverse directly

    return ((e, n), (d, n))

# Compute modular inverse
def mod_inverse(a, m):
    m0, x0, x1 = m, 0, 1
    while a > 1:
        q = a // m
        m, a = a % m, m
        x0, x1 = x1 - q * x0, x0
    return x1 + m0 if x1 < 0 else x1

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
    bit_length = 32  # Adjust this value as needed
    while True:
        attempts += 1
        public_key, private_key = generate_keypair(bit_length)
        hashed_public_key = hash_public_key(public_key)
        if hashed_public_key == target_hash:
            print("\nRSA Key Pair Found:")
            print("Public Key:", public_key)
            print("Private Key:", private_key)
            print("Attempts:", attempts)
            break

        if attempts % 100000 == 0:
            elapsed_time = time.time() - start_time
            speed = attempts / elapsed_time
            print(f"Attempts: {attempts}, Speed: {speed:.2f} keys/s")

if __name__ == "__main__":
    main()
