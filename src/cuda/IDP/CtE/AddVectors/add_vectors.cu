// Courtesy of https://devblogs.nvidia.com/parallelforall/easy-introduction-cuda-c-and-c/ 

#include <iostream>
#include <math.h>
 
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
    if (argc != 2) {
        fprintf(stderr, "usage: add_vectors <problem_size>");
        exit(0);
    }

  int N = 1<<(atoi(argv[1]));
  float *x, *y, *d_x, *d_y;

  cudaMallocHost((void**) &x, N*sizeof(float));
  cudaMallocHost((void**) &y, N*sizeof(float));
 
  cudaMalloc(&d_x, N*sizeof(float));
  cudaMalloc(&d_y, N*sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);
 
  // Launch kernel on 1M elements on the GPU
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  add<<<numBlocks, blockSize>>>(N, d_x, d_y);

  cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost); 

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
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFreeHost(x);
  cudaFreeHost(y);
  
  return 0;
}

