#include <algorithm>
#include <iostream>
#include <vector>
#include <ctime>

__global__ void conv1d(int *image, int *kernel, int *output, const int &IMG_W, const int &K_W) {
    const int tid = blockIdx.x*blockDim.x + threadIdx.x;
    const int R = K_W / 2;
    const int start_idx = tid - R;

    int prod = 0;
    for (int k=0; k<K_W; ++k) {
        if (start_idx + k >= 0 && start_idx + k < IMG_W){
            prod += image[start_idx + k] * kernel[k];
        }
    }
    output[tid] = prod;
}


void conv1d_cpu(std::vector<int> image, std::vector<int> kernel, std::vector<int> &output) {
    const int IMG_W = image.size(), K_W = kernel.size();
    const int R = K_W / 2;
    for (int i=0; i<IMG_W; ++i) {
        const int start_idx = i - R;
        int prod = 0;
        for (int k=0; k<K_W; ++k) {
            if (start_idx + k >= 0 && start_idx + k < IMG_W){
                prod += image[start_idx + k] * kernel[k];
            }
        }
        output[i] = prod;
    }
}

void print_output(std::vector<int> output) {
    std::cout << "output: [";
    for (auto i:output){
        std::cout << i << "\t";  
    }
    std::cout << "]\n";
}

int main() {
    const std::vector<int> image = {1,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5};
    const std::vector<int> kernel = {1,2,1};
    std::vector<int> output = {1,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5};
    std::clock_t start;
    start = std::clock();
    conv1d_cpu(image, kernel, output);
    std::cout << "\nTime (CPU): " << (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;
    print_output(output);
    
    // cuda code
    const int IMG_W = image.size(), K_W = kernel.size();
    const size_t bytes_img = IMG_W * sizeof(int);
    const size_t bytes_k = K_W * sizeof(int);

    // create pointers 
    int *d_img, *d_k, *d_out;
    cudaMalloc(&d_img, bytes_img);
    cudaMalloc(&d_k, bytes_k);
    cudaMalloc(&d_out, bytes_img);
    
    // copy vecs to device
    cudaMemcpy(d_img, image.data(), bytes_img, cudaMemcpyDeviceToHost);
    cudaMemcpy(d_k, kernel.data(), bytes_k, cudaMemcpyDeviceToHost);

    // threads 
    const int NUMBLOCKS = 256, THREADSPERBLOCK = 256;

    // call kernel
    start = std::clock();
    conv1d<<<NUMBLOCKS, THREADSPERBLOCK>>>(d_img, d_k, d_out, IMG_W, K_W);
    std::cout << "\nTime (GPU): " << (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;

    // copy output
    cudaMemcpy(output.data(), d_out, bytes_img, cudaMemcpyHostToDevice);
    print_output(output);
    return 1;
}