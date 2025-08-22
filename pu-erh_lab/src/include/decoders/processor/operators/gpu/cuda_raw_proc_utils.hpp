#pragma once

#define CUDA_CHECK(call)                                                                         \
  do {                                                                                           \
    cudaError_t err = call;                                                                      \
    if (err != cudaSuccess) {                                                                    \
      fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                                                        \
    }                                                                                            \
  } while (0)
