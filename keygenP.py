import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import hashlib

# Define the CUDA kernel code
kernel_code = """
#include <curand_kernel.h>

__global__ void generate_rsa_key_pairs(unsigned char *public_keys, unsigned char *private_keys) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned long long keyspace_start = 20000000000000000;
    unsigned long long keyspace_end = 0x3ffffffffffffffff;
    unsigned char buffer[32];
    curandState state;
    curand_init(clock64(), idx, 0, &state);

    while (1) {
        unsigned long long key = curand(&state) % (keyspace_end - keyspace_start + 1) + keyspace_start;
        sprintf((char *)buffer, "%llx", key);
        unsigned char hash[32];
        sha256(buffer, strlen((char *)buffer), hash);
        // Check if the hash matches the desired public key
        if (memcmp(hash, "13zb1hQbWVsc2S7ZTZnP2G4undNNpdh5so", 32) == 0) {
            memcpy(public_keys, hash, 32);
            // Generate corresponding private key
            unsigned char private_key[32];
            sprintf((char *)private_key, "%llx", key);
            memcpy(private_keys, private_key, 32);
            break;
        }
    }
}

__device__ void sha256(const unsigned char *message, unsigned int len, unsigned char *digest) {
    SHA256_CTX ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, message, len);
    sha256_final(&ctx, digest);
}

__device__ void sha256_init(SHA256_CTX *ctx) {
    ctx->h[0] = 0x6a09e667;
    ctx->h[1] = 0xbb67ae85;
    ctx->h[2] = 0x3c6ef372;
    ctx->h[3] = 0xa54ff53a;
    ctx->h[4] = 0x510e527f;
    ctx->h[5] = 0x9b05688c;
    ctx->h[6] = 0x1f83d9ab;
    ctx->h[7] = 0x5be0cd19;
    ctx->len = 0;
    ctx->tot_len = 0;
}

__device__ void sha256_update(SHA256_CTX *ctx, const unsigned char *message, unsigned int len) {
    unsigned int block_nb;
    unsigned int new_len, rem_len, tmp_len;
    const unsigned char *shifted_message;
    tmp_len = SHA256_BLOCK_SIZE - ctx->len;
    rem_len = len < tmp_len ? len : tmp_len;
    memcpy(&ctx->block[ctx->len], message, rem_len);
    if (ctx->len + len < SHA256_BLOCK_SIZE) {
        ctx->len += len;
        return;
    }
    new_len = len - rem_len;
    block_nb = new_len / SHA256_BLOCK_SIZE;
    shifted_message = message + rem_len;
    sha256_transf(ctx, ctx->block, 1);
    sha256_transf(ctx, shifted_message, block_nb);
    rem_len = new_len % SHA256_BLOCK_SIZE;
    memcpy(ctx->block, &shifted_message[block_nb << 6], rem_len);
    ctx->len = rem_len;
    ctx->tot_len += (block_nb + 1) << 6;
}

__device__ void sha256_final(SHA256_CTX *ctx, unsigned char *digest) {
    unsigned int block_nb;
    unsigned int pm_len;
    unsigned int len_b;
    int i;
    block_nb = (1 + ((SHA256_BLOCK_SIZE - 9) < (ctx->len % SHA256_BLOCK_SIZE)));
    len_b = (ctx->tot_len + ctx->len) << 3;
    pm_len = block_nb << 6;
    memset(ctx->block + ctx->len, 0, pm_len - ctx->len);
    ctx->block[ctx->len] = 0x80;
    SHA2_UNPACK32(len_b, ctx->block + pm_len - 4);
    sha256_transf(ctx, ctx->block, block_nb);
    for (i = 0; i < 8; i++) {
        SHA2_UNPACK32(ctx->h[i], &digest[i << 2]);
    }
}

__device__ void sha256_transf(SHA256_CTX *ctx, const unsigned char *message, unsigned int block_nb) {
    unsigned int w[64];
    unsigned int wv[8];
    unsigned int t1, t2;
    const unsigned char *sub_block;
    int i;
    int j;
    for (i = 0; i < (int)block_nb; i++) {
        sub_block = message + (i << 6);
        for (j = 0; j < 16; j++) {
            SHA2_PACK32(&sub_block[j << 2], &w[j]);
        }
        for (j = 16; j < 64; j++) {
            SHA2_SHR(1, w[j - 2], t1);
            SHA2_SHR(8, w[j - 7], t2);
            SHA2_SHR(7, w[j - 15], t3);
            w[j] = t1 + t2 + t3 + w[j - 16];
        }
        for (j = 0; j < 8; j++) {
            wv[j] = ctx->h[j];
        }
        for (j = 0; j < 64; j++) {
            t1 = wv[7] + SHA2_S1(wv[4]) + SHA2_Ch(wv[4], wv[5], wv[6]) + sha256_k[j] + w[j];
            t2 = SHA2_S0(wv[0]) + SHA2_Maj(wv[0], wv[1], wv[2]);
            wv[7] = wv[6];
            wv[6] = wv[5];
            wv[5] = wv[4];
            wv[4] = wv[3] + t1;
            wv[3] = wv[2];
            wv[2] = wv[1];
            wv[1] = wv[0];
            wv[0] = t1 + t2;
        }
        for (j = 0; j < 8; j++) {
            ctx->h[j] += wv[j];
        }
    }
}
"""

# Compile the kernel code
mod = SourceModule(kernel_code)

# Get the kernel function
generate_rsa_key_pairs = mod.get_function("generate_rsa_key_pairs")

# Define the number of threads per block and the number of blocks
threads_per_block = 256
blocks = 256

# Allocate memory for the output
public_keys = np.zeros((32,), dtype=np.uint8)
private_keys = np.zeros((32,), dtype=np.uint8)

# Call the kernel function
generate_rsa_key_pairs(cuda.Out(public_keys), cuda.Out(private_keys), block=(threads_per_block, 1, 1), grid=(blocks, 1))

# Convert the output to hexadecimal strings
public_key_hex = public_keys.view(np.uint32).astype(np.uint64).tostring().hex()
private_key_hex = private_keys.view(np.uint32).astype(np.uint64).tostring().hex()

# Print the generated key pair
print("Public Key:", public_key_hex)
print("Private Key:", private_key_hex)

