#!/usr/bin/make

# CUDA compiler
NVCC = nvcc
NVCC_FLAGS = -g -Wno-deprecated-gpu-targets

default:
	$(NVCC) $(NVCC_FLAGS) ${KERNEL_DIM} needle.cu -lcudart -DUSIM -o nw_IDP

clean:
	rm -f nw_IDP *.txt gpgpusim_power_report* _cuobjdump_*
