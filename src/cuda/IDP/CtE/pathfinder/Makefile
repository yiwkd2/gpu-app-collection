#!/usr/bin/make

# CUDA compiler
NVCC = nvcc
NVCC_FLAGS = -g -Wno-deprecated-gpu-targets

default:
	$(NVCC) $(NVCC_FLAGS) pathfinder.cu -lcudart -o pathfinder_CtE

clean:
	rm -f pathfinder_CtE *.txt gpgpusim_power_report* _cuobjdump_*
