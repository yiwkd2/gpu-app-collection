#!/usr/bin/make

# CUDA compiler
NVCC = nvcc
NVCC_FLAGS = -g -Wno-deprecated-gpu-targets

default:
	$(NVCC) $(NVCC_FLAGS) pathfinder.cu -lcudart -DUSIM -o pathfinder_IDP

clean:
	rm -f pathfinder_IDP *.txt gpgpusim_power_report* _cuobjdump_*
