#define LIMIT -999
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "needle.h"
#include <cuda.h>
#include <sys/time.h>

// includes, kernels
#include "needle_kernel.cu"

#define USIM
#include "../common.h"

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);


int blosum62[24][24] = {
{ 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0, -2, -1,  0, -4},
{-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, -1,  0, -1, -4},
{-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3,  3,  0, -1, -4},
{-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3,  4,  1, -1, -4},
{ 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4},
{-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2,  0,  3, -1, -4},
{-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
{ 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, -1, -2, -1, -4},
{-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3,  0,  0, -1, -4},
{-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, -3, -3, -1, -4},
{-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, -4, -3, -1, -4},
{-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2,  0,  1, -1, -4},
{-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, -3, -1, -1, -4},
{-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, -3, -3, -1, -4},
{-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, -2, -1, -2, -4},
{ 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2,  0,  0,  0, -4},
{ 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0, -1, -1,  0, -4},
{-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, -4, -3, -2, -4},
{-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, -3, -2, -1, -4},
{ 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, -3, -2, -1, -4},
{-2, -1,  3,  4, -3,  0,  1, -1,  0, -3, -4,  0, -3, -3, -2,  0, -1, -4, -3, -3,  4,  1, -1, -4},
{-1,  0,  0,  1, -3,  3,  4, -2,  0, -3, -3,  1, -1, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
{ 0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2,  0,  0, -2, -1, -1, -1, -1, -1, -4},
{-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,  1}
};

double gettime() {
  struct timeval t;
  gettimeofday(&t,NULL);
  return t.tv_sec+t.tv_usec*1e-6;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{

  printf("WG size of kernel = %d \n", BLOCK_SIZE);

    runTest( argc, argv);

    return EXIT_SUCCESS;
}

void usage(int argc, char **argv)
{
	fprintf(stderr, "Usage: %s <max_rows/max_cols> <penalty> <memory_ratio>\n", argv[0]);
	fprintf(stderr, "\t<dimension>  - x and y dimensions\n");
	fprintf(stderr, "\t<penalty> - penalty(positive integer)\n");
	exit(1);
}

void runTest( int argc, char** argv) 
{
    int max_rows, max_cols, penalty;
    float dev_mem_ratio;
	int *itemsets,  *referrence;
	int size;
	 
    	// the lengths of the two sequences should be able to divided by 16.
	// And at current stage  max_rows needs to equal max_cols
	if (argc == 4)
	{
		max_rows = atoi(argv[1]);
		max_cols = atoi(argv[1]);
		penalty = atoi(argv[2]);
        memory_ratio = atoi(argv[3]);
	}
    else{
        usage(argc, argv);
    }
	
	if(atoi(argv[1])%16!=0){
		fprintf(stderr,"The dimension values must be a multiple of 16\n");
		exit(1);
	}

	max_rows = max_rows + 1;
	max_cols = max_cols + 1;
    
	size = max_cols * max_rows;

    total_malloc += size * sizeof(int);
    total_malloc += size * sizeof(int);
    reserve_gpu_memory();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

	cudaMallocManaged(&referrence, sizeof(int)*size);
	cudaMallocManaged(&itemsets, sizeof(int)*size);
	

	if (!itemsets)
		fprintf(stderr, "error: can not allocate memory");

    	srand ( 7 );
	
	
    	for (int i = 0 ; i < max_cols; i++){
		for (int j = 0 ; j < max_rows; j++){
			itemsets[i*max_cols+j] = 0;
		}
	}
	
	printf("Start Needleman-Wunsch\n");
	
	for( int i=1; i< max_rows ; i++){    //please define your own sequence. 
       		itemsets[i*max_cols] = rand() % 10 + 1;
	}
    	for( int j=1; j< max_cols ; j++){    //please define your own sequence.
       		itemsets[j] = rand() % 10 + 1;
	}


	for (int i = 1 ; i < max_cols; i++){
		for (int j = 1 ; j < max_rows; j++){
			referrence[i*max_cols+j] = blosum62[itemsets[i*max_cols]][itemsets[j]];
		}
	}

    	for( int i = 1; i< max_rows ; i++)
       		itemsets[i*max_cols] = -i * penalty;
	for( int j = 1; j< max_cols ; j++)
       		itemsets[j] = -j * penalty;

#ifdef PREF
	int device = -1;
	cudaGetDevice(&device);
	
	cudaStream_t stream1;
	cudaStreamCreate(&stream1);

	cudaStream_t stream2;
	cudaStreamCreate(&stream2);

	cudaStream_t stream3;
	cudaStreamCreate(&stream3);

	cudaMemPrefetchAsync( referrence, sizeof(int)*size, device, stream1);
	cudaMemPrefetchAsync( itemsets, sizeof(int)*size, device, stream2);
#endif

        dim3 dimGrid;
	dim3 dimBlock(BLOCK_SIZE, 1);
	int block_width = ( max_cols - 1 )/BLOCK_SIZE;

	printf("Processing top-left matrix\n");
    fflush(stdout);
	
	//process top-left matrix
	for( int i = 1 ; i <= block_width ; i++){
		dimGrid.x = i;
		dimGrid.y = 1;
#ifdef PREF
		needle_cuda_shared_1<<<dimGrid, dimBlock, 0, stream3>>>(referrence, itemsets, max_cols, penalty, i, block_width); 
#else
		needle_cuda_shared_1<<<dimGrid, dimBlock>>>(referrence, itemsets, max_cols, penalty, i, block_width); 
        cudaDeviceSynchronize();
#endif
	}
	
	printf("Processing bottom-right matrix\n");
    fflush(stdout);

    	//process bottom-right matrix
	for( int i = block_width - 1  ; i >= 1 ; i--){
		dimGrid.x = i;
		dimGrid.y = 1;
#ifdef PREF
		needle_cuda_shared_2<<<dimGrid, dimBlock, 0, stream3>>>(referrence, itemsets, max_cols, penalty, i, block_width); 
#else
		needle_cuda_shared_2<<<dimGrid, dimBlock>>>(referrence, itemsets, max_cols, penalty, i, block_width);
        cudaDeviceSynchronize();
#endif
	}

    /*
    printf("kernel execution is completed. dump memory, (referrence) addr: %p, size: %lx, "
            "(itemsets) addr: %p, size: %lx\n",
            referrence, sizeof(int)*size, itemsets, sizeof(int)*size);
    fflush(stdout);
    while(1) {}
    */
	
#define TRACEBACK
#ifdef TRACEBACK
	
	FILE *fpo = fopen("result.txt","w");
	fprintf(fpo, "print traceback value GPU:\n");
    
	for (int i = max_rows - 2,  j = max_rows - 2; i>=0, j>=0;){
		int nw, n, w, traceback;
		if ( i == max_rows - 2 && j == max_rows - 2 )
            HOST_ACCESS(READ, &itemsets[ i * max_cols + j]);
			fprintf(fpo, "%d ", itemsets[ i * max_cols + j]); //print the first element
		if ( i == 0 && j == 0 )
           		break;
		if ( i > 0 && j > 0 ){
            HOST_ACCESS(READ, &itemsets[ (i - 1) * max_cols + j - 1 ]);
			nw = itemsets[(i - 1) * max_cols + j - 1];
            HOST_ACCESS(READ, &itemsets[ i * max_cols + j - 1 ]);
		    	w  = itemsets[ i * max_cols + j - 1 ];
            HOST_ACCESS(READ, &itemsets[ (i - 1) * max_cols + j ]);
            		n  = itemsets[(i - 1) * max_cols + j];
		}
		else if ( i == 0 ){
		    	nw = n = LIMIT;
                HOST_ACCESS(READ, &itemsets[ i * max_cols + j - 1 ]);
		    	w  = itemsets[ i * max_cols + j - 1 ];
		}
		else if ( j == 0 ){
		    	nw = w = LIMIT;
                HOST_ACCESS(READ, &itemsets[ (i - 1) * max_cols + j ]);
            	   	n  = itemsets[(i - 1) * max_cols + j];
		}
		else{
		}

		//traceback = maximum(nw, w, n);
		int new_nw, new_w, new_n;
        HOST_ACCESS(READ, &referrence[ i * max_cols + j ]);
		new_nw = nw + referrence[i * max_cols + j];
		new_w = w - penalty;
		new_n = n - penalty;
		
		traceback = maximum(new_nw, new_w, new_n);
		if(traceback == new_nw)
			traceback = nw;
		if(traceback == new_w)
			traceback = w;
		if(traceback == new_n)
            		traceback = n;
			
		fprintf(fpo, "%d ", traceback);

		if(traceback == nw )
			{i--; j--; continue;}

        	else if(traceback == w )
			{j--; continue;}

        	else if(traceback == n )
			{i--; continue;}

		else
		;
	}
	
	fclose(fpo);

#endif

	cudaFree(referrence);
	cudaFree(itemsets);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Elapsed Time: %fms\n", milliseconds);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    MEM_TEST();
}

