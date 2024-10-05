/**
 * 2DConvolution.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <cuda.h>

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

uint64_t NI;
uint64_t NJ;

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void init(DATA_TYPE* A)
{
	int i, j;

	for (i = 0; i < NI; ++i)
    	{
		for (j = 0; j < NJ; ++j)
		{
			A[i*NJ + j] = (float)rand()/RAND_MAX;
        	}
    	}
}

__global__ void Convolution2D_kernel(uint64_t NI, uint64_t NJ, DATA_TYPE *A, DATA_TYPE *B)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

	c11 = +0.2;  c21 = +0.5;  c31 = -0.8;
	c12 = -0.3;  c22 = +0.6;  c32 = -0.9;
	c13 = +0.4;  c23 = +0.7;  c33 = +0.10;

	if ((i < NI-1) && (j < NJ-1) && (i > 0) && (j > 0))
	{
		B[i * NJ + j] =  c11 * A[(i - 1) * NJ + (j - 1)]  + c21 * A[(i - 1) * NJ + (j + 0)] + c31 * A[(i - 1) * NJ + (j + 1)] 
			+ c12 * A[(i + 0) * NJ + (j - 1)]  + c22 * A[(i + 0) * NJ + (j + 0)] +  c32 * A[(i + 0) * NJ + (j + 1)]
			+ c13 * A[(i + 1) * NJ + (j - 1)]  + c23 * A[(i + 1) * NJ + (j + 0)] +  c33 * A[(i + 1) * NJ + (j + 1)];
	}
}


void convolution2DCuda(DATA_TYPE* A, DATA_TYPE* B)
{
	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid((size_t)ceil( ((float)NI) / ((float)block.x) ), (size_t)ceil( ((float)NJ) / ((float)block.y)) );
	
	Convolution2D_kernel<<<grid, block>>>(NI, NJ, A, B);

	// Wait for GPU to finish before accessing on host
	// mock synchronization of memory specific to stream
	cudaDeviceSynchronize();
}


int main(int argc, char *argv[])
{
    if (argc != 2) {
        fprintf(stderr, "usage: 2dconv <problem_size>");
        exit(0);
    }

    NI = atoi(argv[1]);
    NJ = NI;

	DATA_TYPE* A;
	DATA_TYPE* B;

    cudaMallocHost((void**) &A, NI * NJ * sizeof(DATA_TYPE));
    cudaMallocHost((void**) &B, NI * NJ * sizeof(DATA_TYPE));

	//initialize the arrays
	init(A);

    DATA_TYPE* A_cuda;
    DATA_TYPE* B_cuda;

    cudaMalloc((void**) &A_cuda, NI * NJ * sizeof(DATA_TYPE));
    cudaMalloc((void**) &B_cuda, NI * NJ * sizeof(DATA_TYPE));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    cudaMemset(B_cuda, 0, NI * NJ * sizeof(DATA_TYPE));
    cudaMemcpy(A_cuda, A, NI * NJ * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

	convolution2DCuda(A_cuda, B_cuda);

    //cudaMemcpy(B, B_cuda, NI * NJ * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
	
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Elapsed Time: %fms\n", milliseconds);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    //cudaMemcpy(B, B_cuda, NI * NJ * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);

	FILE *fp;

	fp = fopen("result_2DConv.txt","a+");

	for(int i = 0; i < NI*NJ; i+= 10000) {
		fprintf(fp, "%lf\n", B[i]);
	}
	
	fclose(fp);

    cudaFreeHost(A);
    cudaFreeHost(B);

	cudaFree(A_cuda);
	cudaFree(B_cuda);
	
	return 0;
}

