#!/usr/bin/make

# CUDA compiler
NVCC = nvcc
NVCC_FLAGS = -g -Wno-deprecated-gpu-targets

default:
	$(NVCC) $(NVCC_FLAGS) pathfinder.cu -lcudart -o pathfinder_CDP_hw
	$(NVCC) $(NVCC_FLAGS) pathfinder.cu -lcudart -DUSIM -o pathfinder_CDP

clean:
	rm -f pathfinder_CDP_hw pathfinder_CDP *.txt gpgpusim_power_report* _cuobjdump_*
