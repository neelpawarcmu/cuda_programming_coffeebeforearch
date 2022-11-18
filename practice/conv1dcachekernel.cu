// This program implements a 1D convolution using CUDA,
// and stores the kernel in constant memory, and loads
// reused values into shared memory (scratchpad)

# include <iostream>
# include <algorithm>
# include <iostream>
# include <vector>

// !!Diff: store in constant memory
extern __constant__ vector<int> s_array = {1,2,1};
#define K_W = s_array.size();

__global__ void conv1dck(int *image, int *kernel, int *output, const int &K_IMG, const int &K_W) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    // !!Diff: store in shared memory
    extern __shared__ vector<int> s_array;
    int R = K_W / 2;
    int start_idx = tid - R;
    int prod;
    for (int k=0; k<K_W; ++k) {
        if (start_idx + k >= 0 && start_idx + k < s_array.size()) {
            prod += image[start_idx + k] * kernel[k];
    }
    output[tid] = prod;
    }

}

int main() {
    /*
    get image (and kernel? since it is declared in constant mem)
    compute space the vecs will take size_vec * size_int
    
    create *pointers in gpu
    malloc at &pointers for the vecs

    memcpy vecs from host to device
    fn<<<>>>(args)
    
    memcpy output vec from device to host
    */

    const int IMG_H = image.size(), K_H = kernel.size();
    const int bytes_image = IMG_H * sizeof(int);
    const int bytes_kernel = K_H * sizeof(int);
    const int *d_image, *d_kernel, *d_out;
    
    cudaMalloc(&d_image, bytes_image);
    cudaMalloc(&d_kernel, bytes_kernel);
    cudaMalloc(&d_out, bytes_image);
    std::vector<int> h_result(IMG_H);

    cudaMemcpy(d_image, image, bytes_image, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, bytes_kernel, cudaMemcpyHostToDevice);


    const int THREADSPERBLOCK = 256, NUMBLOCKS = IMG_H / THREADSPERBLOCK;
    conv1dck<<<THREADSPERBLOCK, NUMBLOCKS>>>(d_image, d_kernel, d_out);

    cudaMemcpy(h_result.data(), d_result, bytes_image, cudaMemcpyDeviceToHost);

    cudaFree(d_image);
    cudaFree(d_kernel);
    cudaFree(d_out);
    delete[] h_result;

    return 0;
}