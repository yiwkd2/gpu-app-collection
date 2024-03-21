#!/usr/bin/make

# CUDA compiler
NVCC = nvcc
NVCC_FLAGS = -g -Wno-deprecated-gpu-targets

default:
	$(NVCC) $(NVCC_FLAGS) bfs.cu -lcudart -o bfs_CtE

clean:
	rm -f bfs_CtE *.linkinfo *.txt gpgpusim_power_report* _cuobjdump_*
