#!/usr/bin/make

# CUDA compiler
NVCC = nvcc
NVCC_FLAGS = -g -Wno-deprecated-gpu-targets

SRC = hotspot.cu

default:
	$(NVCC) $(NVCC_FLAGS) $(KERNEL_DIM) hotspot.cu -lcudart -o hotspot_CtE

clean:
	rm -f hotspot_CtE *.txt gpgpusim_power_report* _cuobjdump_*
