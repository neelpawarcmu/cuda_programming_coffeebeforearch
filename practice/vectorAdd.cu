
__global__ void vectorAdd(float *A, float *B, float *C, int N) {
    const int idx = threadIdx.x;
    if (idx < N){
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    const int N = 1 << 16;
    vector<int> A(N);
    vector<int> B(N);
    vector<int> C(N);

    // Initialize random numbers in each array
    for (int i = 0; i < N; i++) {
        A[idx] = rand() % 100;
        B[idx] = rand() % 100;
    }

    // allocate memory
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // copy tensors
    cudaMemcpy(&d_a, A.data(), cudaMemcpyHostToDevice);
}