#!/usr/bin/make

# CUDA compiler
NVCC = nvcc
NVCC_FLAGS = -g -Wno-deprecated-gpu-targets

default:
	$(NVCC) $(NVCC_FLAGS) bfs.cu -lcudart -DUSIM -o bfs_IDP

clean:
	rm -f bfs_IDP *.linkinfo *.txt gpgpusim_power_report* _cuobjdump_*
