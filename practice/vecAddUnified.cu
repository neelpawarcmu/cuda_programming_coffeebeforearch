/*
cudaMallocManaged

replaces cudaMalloc, has same args
removes need of any memcpy statements
just needs a cudaDeviceSynchronize() after <<<>>>
*/
#include <stdlib.h>     /* srand, rand */
#include <string>     
#include <vector>     
#include <iostream>     /* cout */

__global__ void vectorAdd(int *A, int *B, int *C, int N) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N){
        C[idx] =A[idx] + B[idx];
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
    // std::vector<int>A = create_vec(N);
    // std::vector<int> B = create_vec(N);
    // std::vector<int> C = create_vec(N);

    const int bytes = N * sizeof(int);

    // malloc
    int *A, *B, *C;
    cudaMallocManaged(&A, bytes);
    cudaMallocManaged(&B, bytes);
    cudaMallocManaged(&C, bytes);

    // Initialize vectors
    for (int i = 0; i < N; i++) {
        A[i] = rand() % 100;
        B[i] = rand() % 100;
    }

    // memcpy not needed

    // kernel
    const int NUMTHREADS = 1024; // per block
    const int NUMBLOCKS = N / NUMTHREADS + 1;
    vectorAdd<<<NUMBLOCKS, NUMTHREADS>>>(A, B, C, N);

    // memcpy back not needed
    cudaDeviceSynchronize();
    
    // free mem still needed
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    print_vec(std::vector<int>A, "A");
    print_vec(B, "B");
    print_vec(C, "C");
    return 1;
}