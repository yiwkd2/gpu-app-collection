// Courtesy of https://devblogs.nvidia.com/parallelforall/easy-introduction-cuda-c-and-c/ 

#include <iostream>
#include <math.h>

#include "../../common.h"
 
// CUDA kernel to add elements of two arrays
__global__
void add(int n, float *x, float *y)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    y[i] = x[i] + y[i];
}
 
int main(int argc, char *argv[])
{
    if (argc != 3) {
        fprintf(stderr, "usage: add_vectors <problem_size> <working_set_ratio>");
        exit(0);
    }
  int N = 1<<(atoi(argv[1]));
  working_set_ratio = atoi(argv[2]);

  total_malloc += N*sizeof(float);
  total_malloc += N*sizeof(float);
  reserve_gpu_memory();

  float* x;
  float* y;

  cudaMallocManaged( &x, N&sizeof(float));
  cudaMallocManaged( &y, N&sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    HOST_ACCESS(WRITE, &x[i]);
    y[i] = 2.0f;
    HOST_ACCESS(WRITE, &y[i]);
  }

  cudaMakeManagedByDevice(x);
  cudaMakeManagedByDevice(y);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
 
  // Launch kernel on 1M elements on the GPU
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  add<<<numBlocks, blockSize>>>(N, x, y);

	cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Elapsed Time: %fms\n", milliseconds);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
 
  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;
 
  // Free memory
  cudaFree(x);
  printf("free x\n");
  cudaFree(y);
  printf("free y\n");

  MEM_TEST();
  
  return 0;
}

