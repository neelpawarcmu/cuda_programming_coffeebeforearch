### Boilerplate for naive image convolution, 1d

0. get image and kernel  find sizes
1. create output of calculated size: `vector<int> h_out`
2. compute bytes the 3 vecs will take: `size_vec * sizeof(int)`

3. create pointers in gpu: `int *d_img, *d_kernel, *d_out`
    malloc at &pointers for the vecs: `cudaMalloc(&d_img, bytes)`

4. memcpy(to, from, bytes, devices) vecs from host to device: 
    `cudaMemcpy(d_img, image.data(), bytes, cudaMemcpyDeviceToHost)`

5. ThreadsPerBlock = 256, BlockDim = image_size / ThreadsPerBlock + 1
    `fn<<<BlockDim, ThreadsPerBlock>>>(args)`

6. memcpy output vec from device to host: 
    `cudaMemcpy(result.data(), d_out, bytes, cudaMemcpyHostToDevice)`

7. cudaFree all declared in malloc: `cudaFree(d_img)`
    delete cpu vecs if necessary: `delete[] h_out`