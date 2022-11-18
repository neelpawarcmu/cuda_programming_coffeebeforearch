### Boilerplate for naive 1d CUDA image processing code

kernel: `__global__ void fn_name(int *image, int *filter, int *output)` 
1. get thread id: `int tid = blockDim.x * blockIdx.x + threadIdx.x`
2. main logic: 
    `prod = iterate over kernel elements and multiply with image`
    `output[tid] = prod` 


main fn:

0. get image and filter, find sizes
1. create output of calculated size: `vector<int> h_out`
2. compute bytes the 3 vecs will take: `size_vec * sizeof(int)`

3. create pointers in gpu: `int *d_img, *d_filter, *d_out`
    malloc at &pointers for the vecs: `cudaMalloc(&d_img, bytes)`

4. memcpy(to, from, bytes, devices) vecs from host to device: 
    `cudaMemcpy(d_img, image.data(), bytes, cudaMemcpyDeviceToHost)`

5. ThreadsPerBlock = 256, BlockDim = image_size / ThreadsPerBlock + 1
    `fn<<<BlockDim, ThreadsPerBlock>>>(image, filter)`

6. memcpy output vec from device to host: 
    `cudaMemcpy(result.data(), d_out, bytes, cudaMemcpyHostToDevice)`

7. cudaFree all declared in malloc: `cudaFree(d_img)`
    delete cpu vecs if necessary: `delete[] h_out`