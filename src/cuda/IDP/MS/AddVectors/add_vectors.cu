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
    if (argc != 3) {
        fprintf(stderr, "usage: add_vectors <problem_size> <num_streams>");
        exit(0);
    }

  int N = 1<<(atoi(argv[1]));
  int num_streams = atoi(argv[2]);

  cudaStream_t stream[num_streams];
  float* x[num_streams];
  float* y[num_streams];
  float* d_x[num_streams];
  float* d_y[num_streams];

  for (int i = 0; i < num_streams; i++)
    cudaStreamCreate(&stream[i]); 
  
  cudaMallocHost((void**) &x[0], N * sizeof(float));
  cudaMallocHost((void**) &y[0], N * sizeof(float));

  cudaMalloc(&d_x[0], N * sizeof(float));
  cudaMalloc(&d_y[0], N * sizeof(float));

  float* tmp_x = x[0];
  float* tmp_y = y[0];
  float* tmp_d_x = d_x[0];
  float* tmp_d_y = d_y[0];
  for (int i = 0; i < N; i++) {
    tmp_x[i] = 1.0f;
    tmp_y[i] = 2.0f;
  }

  for (int i = 1; i < num_streams; i++) {
    x[i] = &tmp_x[i * (N / num_streams)];
    y[i] = &tmp_y[i * (N / num_streams)];
    d_x[i] = &tmp_d_x[i * (N / num_streams)];
    d_y[i] = &tmp_d_y[i * (N / num_streams)];
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  for (int i = 0; i < num_streams; i++) {
    cudaMemcpyAsync(d_x[i], x[i], (N / num_streams) * sizeof(float), cudaMemcpyHostToDevice, stream[i]);
    cudaMemcpyAsync(d_y[i], y[i], (N / num_streams) * sizeof(float), cudaMemcpyHostToDevice, stream[i]);

    int blockSize = 256;
    int numBlocks = ((N / num_streams) + blockSize - 1) / blockSize;
    add<<<numBlocks, blockSize, 0, stream[i]>>>(N / num_streams, d_x[i], d_y[i]);

    cudaMemcpyAsync(y[i], d_y[i], (N / num_streams) * sizeof(float), cudaMemcpyDeviceToHost, stream[i]);
  }

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
    maxError = fmax(maxError, fabs(tmp_y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;
 
  // Free memory
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFreeHost(x);
  cudaFreeHost(y);
  
  return 0;
}

