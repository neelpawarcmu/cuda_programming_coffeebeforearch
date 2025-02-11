#include <stdlib.h>     /* srand, rand */
#include <string>     
#include <vector>     
#include <iostream>     /* cout */

__global__ void vectorAdd(int *A, int *B, int *C, int N) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N){
        C[idx] = A[idx] + B[idx];
    }
}

std::vector<int> create_vec(const int N){
    std::vector<int> vec(N);
    for (int i=0; i<N; ++i) {
        // Initialize random numbers in array
        vec[i] = rand() %100;
    }
    return vec;
}

void print_vec(std::vector<int> vec, std::string vec_name) {
    std::cout << vec_name << ": [";
    for (auto i:vec){
        std::cout << i << "\t";  
    }
    std::cout << "]\n";
}

int main() {
    const int N = 16;
    std::vector<int> A = create_vec(N);
    std::vector<int> B = create_vec(N);
    std::vector<int> C = create_vec(N);

    const int bytes = N * sizeof(int);

    // malloc
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // memcpy
    cudaMemcpy(d_a, A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, B.data(), bytes, cudaMemcpyHostToDevice);

    // kernel
    const int NUMTHREADS = 256; // per block
    const int NUMBLOCKS = N / NUMTHREADS + 1;
    // dim3 block_dim(NUMTHREADS, NUMTHREADS);
    // dim3 grid_dim(NUMBLOCKS, NUMBLOCKS);
    vectorAdd<<<NUMBLOCKS, NUMTHREADS>>>(d_a, d_b, d_c, N);

    // memcpy back
    cudaMemcpy(C.data(), d_c, bytes, cudaMemcpyHostToDevice);

    print_vec(A, "A");
    print_vec(B, "B");
    print_vec(C, "C");
    return 1;
}