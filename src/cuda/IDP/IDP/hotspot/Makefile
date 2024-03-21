#!/usr/bin/make

# CUDA compiler
NVCC = nvcc
NVCC_FLAGS = -g -Wno-deprecated-gpu-targets

SRC = hotspot.cu

default:
	$(NVCC) $(NVCC_FLAGS) $(KERNEL_DIM) hotspot.cu -lcudart -DUSIM -o hotspot_IDP

clean:
	rm -f hotspot_IDP *.txt gpgpusim_power_report* _cuobjdump_*
