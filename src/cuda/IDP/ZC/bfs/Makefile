#!/usr/bin/make

# CUDA compiler
NVCC = nvcc
NVCC_FLAGS = -g -Wno-deprecated-gpu-targets

default:
	$(NVCC) $(NVCC_FLAGS) bfs.cu -lcudart -o bfs_ZC

clean:
	rm -f bfs_ZC *.linkinfo *.txt gpgpusim_power_report* _cuobjdump_*
