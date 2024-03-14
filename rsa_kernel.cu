#include <curand_kernel.h>
#include <stdint.h>

extern "C" {
__global__ void generate_rsa_key_pairs(uint64_t* output, uint64_t min_key, uint64_t max_key) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(clock64() + idx, 0, 0, &state); // Initialize random number generator

    while (1) {
        uint64_t n = curand(&state) % (max_key - min_key) + min_key;
        uint64_t e = curand(&state) % (max_key - min_key) + min_key;
        uint64_t d = curand(&state) % (max_key - min_key) + min_key;

        output[3 * idx] = n;
        output[3 * idx + 1] = e;
        output[3 * idx + 2] = d;

        if (output[3 * idx] == 0x13zb1hQbWVsc2S7ZTZnP2G4undNNpdh5soULL) // Check if public key matches
            break;
    }
}
}
