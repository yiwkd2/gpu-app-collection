#include <cuda_runtime.h>
#include <cstdint>

#define READ NULL
#define WRITE ((cudaIpcMemHandle_t*) 1)

#ifdef USIM
#define HOST_ACCESS(access_type, vaddr) cudaIpcGetMemHandle(access_type, vaddr)
#define cudaMakeManagedByDevice(vaddr) cudaIpcGetMemHandle((cudaIpcMemHandle_t*) 2, vaddr)
#define MEM_TEST() cudaIpcGetMemHandle((cudaIpcMemHandle_t*) 3, 0)
#else
#define INIT() do {} while (0)
#define HOST_ACCESS(...) do {} while (0)
#define cudaMakeManagedByDevice(...) do {} while (0)
#define MEM_TEST() do {} while (0)
#endif

unsigned working_set_ratio = 100;
size_t total_malloc = 0;
size_t free_memory, total_memory;

FILE* fapp_trace = fopen("app_trace.txt", "w");

void reserve_gpu_memory() {
    void* dummy;
    uint64_t extra_malloc_size;
    printf("(total malloc) %fMB\n", (float) total_malloc / 1024 / 1024);
    if (working_set_ratio > 100) {
        cudaMemGetInfo(&free_memory, &total_memory);
        extra_malloc_size = free_memory -
            (uint64_t) (total_malloc * 100 / (float) working_set_ratio);

        printf("(before malloc) working set ratio: %f%%\n",
                (float) total_malloc / free_memory * 100);

        cudaError_t status = cudaMalloc(&dummy, extra_malloc_size);
        if (status != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(status));
            fflush(stderr);
            exit(0);
        }
        cudaMemGetInfo(&free_memory, &total_memory);
        printf("(after malloc) working set ratio: %f%%\n",
                (float) total_malloc / free_memory * 100);
    }
    fflush(stdout);
}
