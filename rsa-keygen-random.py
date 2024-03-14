import time
import random
from Crypto.PublicKey import RSA
from multiprocessing import Pool, cpu_count

# Define the range for RSA key generation
min_key = 20000000000000000
max_key = 0x3ffffffffffffffff

# Define the target public key
target_public_key = "13zb1hQbWVsc2S7ZTZnP2G4undNNpdh5so"

# Function to generate random RSA key pairs
def generate_rsa_key_pair(dummy):
    while True:
        key_length = random.randint(1024, 4096)  # Choose a random key length
        key = RSA.generate(key_length)
        if min_key <= key.n <= max_key:
            return key, key.publickey().export_key().decode("utf-8")

# Function to check if a public key matches the target
def is_target_public_key(public_key):
    return public_key == target_public_key

if __name__ == '__main__':
    start_time = time.time()

    # Create a pool of worker processes
    pool = Pool(cpu_count())

    # Generate RSA key pairs concurrently
    found = False
    iteration = 0
    while not found:
        iteration += 1
        results = pool.map(generate_rsa_key_pair, range(cpu_count()))
        for rsa_key, rsa_public_key in results:
            print(f"Iteration: {iteration}, Public Key: {rsa_public_key}")
            if is_target_public_key(rsa_public_key):
                found = True
                break

    pool.close()
    pool.join()

    elapsed_time = time.time() - start_time

    print("RSA key pair found!")
    print(f"Private key: {rsa_key.export_key().decode('utf-8')}")
    print(f"Public key: {rsa_public_key}")
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
