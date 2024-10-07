#!/usr/bin/make

# CUDA compiler
NVCC = nvcc
NVCC_FLAGS = -g -Wno-deprecated-gpu-targets

default:
	$(NVCC) $(NVCC_FLAGS) bfs.cu -lcudart -o bfs_CDP_hw
	$(NVCC) $(NVCC_FLAGS) bfs.cu -lcudart -DPREFETCH -o bfs_CDP_prefetch_hw
	$(NVCC) $(NVCC_FLAGS) bfs.cu -lcudart -DUSIM -o bfs_CDP

clean:
	rm -f bfs_CDP_hw bfs_CDP_prefetch_hw bfs_CDP *.linkinfo *.txt gpgpusim_power_report* _cuobjdump_*
