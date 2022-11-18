/*
some issue here, output doesnt update
*/

#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <vector>
#include <stdlib.h>     /* srand, rand */
#include <string>

__global__ void conv2d( int *image, int *filter, int *output, 
                        const int &IMG_H, const int &IMG_W, 
                        const int &F_H, const int &F_W, 
                        const int &OUT_H, const int &OUT_W) {
    // init thread id
    const int t_h = blockDim.y * blockIdx.y + threadIdx.y; // thread row ie. height
    const int t_w = blockDim.x * blockIdx.x + threadIdx.x; // thread col ie. width
    const int R_W = F_W / 2, R_H = F_H / 2;

    const int start_idx_h = t_h - R_H;
    const int start_idx_w = t_w - R_W;

    int prod = 0;
    for (int h=0; h<F_H; ++h) {
        for (int w=0; w<F_W; ++w) {
            if (start_idx_h + h >= 0 && start_idx_h + h < IMG_H &&
                start_idx_w + w >= 0 && start_idx_w + w < IMG_W) {
                    prod += (image[(start_idx_h + h)*IMG_W + (start_idx_w + w)] * filter[h*F_W + w]);
            }
        }
    }
    output[t_h*OUT_W + t_w] = prod;
}

void print_matrix(std::vector<std::vector<int>> &matrix, std::string matrix_name) {
    std::cout << "\n" << matrix_name << ":\n";
    for (int i=0; i<matrix.size(); ++i) {
        for (int j=0; j<matrix[0].size(); ++j) {
            std::cout << matrix[i][j] << "\t";
        }
        std::cout << "\n";
    }
}

std::vector<std::vector<int>> main_fn(std::vector<std::vector<int>> &image,  std::vector<std::vector<int>> &filter) {
    // find sizes
    const int IMG_H = image.size(), IMG_W = image[0].size(), F_H = filter.size(), F_W = filter[0].size();
    
    // create output vector o = (i - f + 2p) / s + 1
    const int OUT_H = IMG_H, OUT_W = IMG_W; // (IMG_H - F_H) + 1, OUT_W = (IMG_W - F_W) + 1;
    std::vector<std::vector<int>> output(OUT_H, std::vector<int>(OUT_W, 0));

    // malloc
    int bytes_image = IMG_H * IMG_W * sizeof(int);
    int bytes_filter = F_H * F_W * sizeof(int);
    int bytes_output = OUT_H * OUT_W * sizeof(int);
    int *d_image, *d_filter, *d_output;
    cudaMallocManaged(&d_image, bytes_image);
    cudaMallocManaged(&d_filter, bytes_filter);
    cudaMallocManaged(&d_output, bytes_output);

    // memcpy
    // cudaMemcpy(d_image, image.data(), bytes_image, cudaMemcpyDeviceToHost);
    // cudaMemcpy(d_filter, filter.data(), bytes_filter, cudaMemcpyDeviceToHost);

    // call kernel
    const int NUMTHREADS = 16;
    const int NUMBLOCKS = (IMG_H * IMG_W) / NUMTHREADS + 1;
    dim3 block_dim(NUMTHREADS, NUMTHREADS);
    dim3 grid_dim(NUMBLOCKS, NUMBLOCKS);
    conv2d<<<grid_dim, block_dim>>>(d_image, d_filter, d_output, IMG_H, IMG_W, F_H, F_W, OUT_H, OUT_W);
    
    // memcpy back
    // cudaMemcpy(output.data(), d_output, bytes_output, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // free cuda mem
    cudaFree(d_image);
    cudaFree(d_filter);
    cudaFree(d_output);
    return output;
}

int main() {
    int IMG_H = 1 << 3, IMG_W = 1 << 4,  F_H = 2, F_W = 3;
    std::vector<std::vector<int>> image(IMG_H, std::vector<int>(IMG_W, 0));
    std::vector<std::vector<int>> filter(F_H, std::vector<int>(F_W, 0));
    
    for (int i=0; i<IMG_H; ++i) {
        for (int j=0; j<IMG_W; ++j) {
            int num = rand()%100;
            image[i][j] = num;
        }
    }
    for (int i=0; i<F_H; ++i) {
        for (int j=0; j<F_W; ++j) {
            int num = rand()%5;
            filter[i][j] = num;
        }
    }

    std::vector<std::vector<int>> out = main_fn(image, filter);
    print_matrix(image, "Img");
    print_matrix(filter, "Filter");
    print_matrix(out, "Output");

    return 0;
}