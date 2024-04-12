/**
 * fdtd2d.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include <cuda.h>

//#include "../../common/polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 10.05

#define GPU_DEVICE 0

unsigned tmax;
uint64_t NX;
uint64_t NY;


/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

#include "../../common.h"

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;



void init_arrays(DATA_TYPE* _fict_, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz)
{
	int i, j;

  	for (i = 0; i < tmax; i++)
	{
		_fict_[i] = (DATA_TYPE) i;
	}
	
	for (i = 0; i < NX; i++)
	{
		for (j = 0; j < NY; j++)
		{
			ex[i*NY + j] = ((DATA_TYPE) i*(j+1) + 1) / NX;
            HOST_ACCESS(WRITE, &ex[i*NY + j]);
			ey[i*NY + j] = ((DATA_TYPE) (i-1)*(j+2) + 2) / NX;
            HOST_ACCESS(WRITE, &ey[i*NY + j]);
			hz[i*NY + j] = ((DATA_TYPE) (i-9)*(j+4) + 3) / NX;
            HOST_ACCESS(WRITE, &hz[i*NY + j]);
		}
	}
}


void runFdtd(DATA_TYPE* _fict_, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz)
{
	int t, i, j;
	
	for (t=0; t < tmax; t++)  
	{
		for (j=0; j < NY; j++)
		{
			ey[0*NY + j] = _fict_[t];
		}
	
		for (i = 1; i < NX; i++)
		{
       		for (j = 0; j < NY; j++)
			{
       			ey[i*NY + j] = ey[i*NY + j] - 0.5*(hz[i*NY + j] - hz[(i-1)*NY + j]);
        		}
		}

		for (i = 0; i < NX; i++)
		{
       		for (j = 1; j < NY; j++)
			{
				ex[i*(NY+1) + j] = ex[i*(NY+1) + j] - 0.5*(hz[i*NY + j] - hz[i*NY + (j-1)]);
			}
		}

		for (i = 0; i < NX; i++)
		{
			for (j = 0; j < NY; j++)
			{
				hz[i*NY + j] = hz[i*NY + j] - 0.7*(ex[i*(NY+1) + (j+1)] - ex[i*(NY+1) + j] + ey[(i+1)*NY + j] - ey[i*NY + j]);
			}
		}
	}
}

/*
void compareResults(DATA_TYPE* hz1, DATA_TYPE* hz2)
{
	int i, j, fail;
	fail = 0;
	
	for (i=0; i < NX; i++) 
	{
		for (j=0; j < NY; j++) 
		{
			if (percentDiff(hz1[i*NY + j], hz2[i*NY + j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
			{
				fail++;
			}
		}
	}
	
	// Print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}


void GPU_argv_init()
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
	printf("setting device %d with name %s\n",GPU_DEVICE,deviceProp.name);
	cudaSetDevice( GPU_DEVICE );
}
*/


__global__ void fdtd_step1_kernel(uint64_t NX, uint64_t NY, DATA_TYPE* _fict_, DATA_TYPE *ex, DATA_TYPE *ey, DATA_TYPE *hz, int t)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < NX) && (j < NY))
	{
		if (i == 0) 
		{
			ey[i * NY + j] = _fict_[t];
		}
		else
		{ 
			ey[i * NY + j] = ey[i * NY + j] - 0.5f*(hz[i * NY + j] - hz[(i-1) * NY + j]);
		}
	}
}



__global__ void fdtd_step2_kernel(uint64_t NX, uint64_t NY, DATA_TYPE *ex, DATA_TYPE *ey, DATA_TYPE *hz, int t)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	
	if ((i < NX) && (j < NY) && (j > 0))
	{
		ex[i * (NY+1) + j] = ex[i * (NY+1) + j] - 0.5f*(hz[i * NY + j] - hz[i * NY + (j-1)]);
	}
}


__global__ void fdtd_step3_kernel(uint64_t NX, uint64_t NY, DATA_TYPE *ex, DATA_TYPE *ey, DATA_TYPE *hz, int t)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	
	if ((i < NX) && (j < NY))
	{	
		hz[i * NY + j] = hz[i * NY + j] - 0.7f*(ex[i * (NY+1) + (j+1)] - ex[i * (NY+1) + j] + ey[(i + 1) * NY + j] - ey[i * NY + j]);
	}
}


void fdtdCuda(DATA_TYPE* _fict_, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz)//, DATA_TYPE* hz_outputFromGpu)
{
	//double t_start, t_end;
/*
	DATA_TYPE *_fict_gpu;
	DATA_TYPE *ex_gpu;
	DATA_TYPE *ey_gpu;
	DATA_TYPE *hz_gpu;

	cudaMalloc((void **)&_fict_gpu, sizeof(DATA_TYPE) * tmax);
	cudaMalloc((void **)&ex_gpu, sizeof(DATA_TYPE) * NX * (NY + 1));
	cudaMalloc((void **)&ey_gpu, sizeof(DATA_TYPE) * (NX + 1) * NY);
	cudaMalloc((void **)&hz_gpu, sizeof(DATA_TYPE) * NX * NY);

	cudaMemcpy(_fict_gpu, _fict_, sizeof(DATA_TYPE) * tmax, cudaMemcpyHostToDevice);
	cudaMemcpy(ex_gpu, ex, sizeof(DATA_TYPE) * NX * (NY + 1), cudaMemcpyHostToDevice);
	cudaMemcpy(ey_gpu, ey, sizeof(DATA_TYPE) * (NX + 1) * NY, cudaMemcpyHostToDevice);
	cudaMemcpy(hz_gpu, hz, sizeof(DATA_TYPE) * NX * NY, cudaMemcpyHostToDevice);
*/
	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid( (size_t)ceil(((float)NY) / ((float)block.x)), (size_t)ceil(((float)NX) / ((float)block.y)));

	//t_start = rtclock();

	for(int t = 0; t< tmax; t++)
	{
		fdtd_step1_kernel<<<grid,block>>>(NX, NY, _fict_, ex, ey, hz, t);
		fdtd_step2_kernel<<<grid,block>>>(NX, NY, ex, ey, hz, t);
		fdtd_step3_kernel<<<grid,block>>>(NX, NY, ex, ey, hz, t);
	}

    cudaDeviceSynchronize();
	
	//t_end = rtclock();
    	//fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

	//cudaMemcpy(hz_outputFromGpu, hz_gpu, sizeof(DATA_TYPE) * NX * NY, cudaMemcpyDeviceToHost);	
		
	//cudaFree(_fict_gpu);
	//cudaFree(ex_gpu);
	//cudaFree(ey_gpu);
	//cudaFree(hz_gpu);
}


int main(int argc, char *argv[])
{
    if (argc != 4) {
        fprintf(stderr, "usage: fdtd2d <tmax> <problem_size> <working_set_ratio>\n");
        exit(0);
    }
    tmax = atoi(argv[1]);
    NX = atoi(argv[2]);
    NY = NX;
    working_set_ratio = atoi(argv[3]);

	//double t_start, t_end;

	DATA_TYPE* _fict_;
	DATA_TYPE* ex;
	DATA_TYPE* ey;
	DATA_TYPE* hz;
	//DATA_TYPE* hz_outputFromGpu;

/*
	_fict_ = (DATA_TYPE*)malloc(tmax*sizeof(DATA_TYPE));
	ex = (DATA_TYPE*)malloc(NX*(NY+1)*sizeof(DATA_TYPE));
	ey = (DATA_TYPE*)malloc((NX+1)*NY*sizeof(DATA_TYPE));
	hz = (DATA_TYPE*)malloc(NX*NY*sizeof(DATA_TYPE));
*/	
    total_malloc += tmax*sizeof(DATA_TYPE);
    total_malloc += NX*(NY+1)*sizeof(DATA_TYPE);
    total_malloc += (NX+1)*NY*sizeof(DATA_TYPE);
    total_malloc += NX*NY*sizeof(DATA_TYPE);
    reserve_gpu_memory();

	cudaMallocManaged(&_fict_, tmax*sizeof(DATA_TYPE));
    memset(_fict_, 0, tmax*sizeof(DATA_TYPE));
    printf("alloc _fict_, size: %lu\n", tmax*sizeof(DATA_TYPE));
	cudaMallocManaged(&ex, NX*(NY+1)*sizeof(DATA_TYPE));
    memset(ex, 0, NX*(NY+1)*sizeof(DATA_TYPE));
    printf("alloc ex, size: %lu\n", NX*(NY+1)*sizeof(DATA_TYPE));
	cudaMallocManaged(&ey, (NX+1)*NY*sizeof(DATA_TYPE));
    memset(ey, 0, (NX+1)*NY*sizeof(DATA_TYPE));
    printf("alloc ey, size: %lu\n", (NX+1)*NY*sizeof(DATA_TYPE));
	cudaMallocManaged(&hz, NX*NY*sizeof(DATA_TYPE));
    memset(hz, 0, NX*NY*sizeof(DATA_TYPE));
    printf("alloc hz, size: %lu\n", NX*NY*sizeof(DATA_TYPE));
	
	//hz_outputFromGpu = (DATA_TYPE*)malloc(NX*NY*sizeof(DATA_TYPE));

	init_arrays(_fict_, ex, ey, hz);

    cudaMakeManagedByDevice(_fict_);
    cudaMakeManagedByDevice(ex);
    cudaMakeManagedByDevice(ey);
    cudaMakeManagedByDevice(hz);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

	//GPU_argv_init();
	fdtdCuda(_fict_, ex, ey, hz);//, hz_outputFromGpu);

    /*
    // emulate host access
    double dummy;
	for(int i = 0; i < NX*NY; i+= 1000) {
        HOST_ACCESS(READ, &hz[i]);
        dummy = hz[i];
	}
    */

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Elapsed Time: %fms\n", milliseconds);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

	//t_start = rtclock();
	//runFdtd(_fict_, ex, ey, hz);
	//t_end = rtclock();
	
	//fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);
	
	//compareResults(hz, hz_outputFromGpu);

	printf("-------------Size: %lf--------------\n", 
		(float)(sizeof(DATA_TYPE)*(tmax + NX*(NY+1)+(NX+1)*NY+NX*NY))/1024.0/1024.0);
	FILE *fp;

	fp = fopen("file.txt","w");

	for(int i = 0; i < NX*NY; i+= 1000) {
		fprintf(fp, "%lf\n", hz[i]);
	}
	
	fclose(fp);

	cudaFree(_fict_);
    printf("free _fict_\n");
	cudaFree(ex);
    printf("free ex\n");
	cudaFree(ey);
    printf("free ey\n");
	cudaFree(hz);
    printf("free hz\n");
	//free(hz_outputFromGpu);

    MEM_TEST();    
 
	return 0;
}

