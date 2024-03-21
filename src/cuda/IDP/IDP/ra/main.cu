#include <iostream>
#include <math.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define USIM
#include "../common.h"

uint64_t N;

__global__ void kernel(float* input, float* output, float* table, size_t size)
{
	int x_id = blockIdx.x * blockDim.x + threadIdx.x;
	if (x_id > size || x_id % 100 != 0)
    		return;

	float in_f = input[x_id];
	int in_i = (int)(floor(in_f));
	int table_index = (int)((in_f - float(in_i)) *( (float)(size) ));
	float* t = table + table_index;
	output[table_index] = t[0] * in_f;
}

int main(int argc, char *argv[])
{
    if (argc != 3) {
        fprintf(stderr, "usage: ra <exp> <memory_ratio>\n");
        exit(0);
    }
    N = atoi(argv[1]);
    N = (1 << N);
    memory_ratio = atoi(argv[2]);

  float *input, *output, *table;

  total_malloc += N*sizeof(float);
  total_malloc += N*sizeof(float);
  total_malloc += N*sizeof(float);
  reserve_gpu_memory();

  cudaHostAlloc(&input, N*sizeof(float), 0);
  cudaHostAlloc(&output, N*sizeof(float), 0);
  cudaHostAlloc(&table, N*sizeof(float), 0);

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    input[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    //printf("-%lf-\n", input[i]);
    table[i] = ((float)(i));
  }

  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;

  MAKE_MANAGED(input);
  MAKE_MANAGED(output);
  MAKE_MANAGED(table);

  kernel<<<numBlocks, blockSize>>>(input, output, table, N);

  cudaDeviceSynchronize();

  MAKE_UNMANAGED(output);

  printf("print result\n");
  for (int i = 0; i < N; i++)
    if(output[i] != 0) {
  	printf("-%d %lf-\n", i, output[i]);
    }

  cudaFree(input);
  cudaFreeHost(output);
  cudaFree(table);

  MEM_TEST();

  return 0;
}
