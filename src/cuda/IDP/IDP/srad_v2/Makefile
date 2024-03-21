#!/usr/bin/make

# CUDA compiler
NVCC = nvcc
NVCC_FLAGS = -g -Wno-deprecated-gpu-targets

default:
	$(NVCC) $(NVCC_FLAGS) $(KERNEL_DIM) srad.cu -lcudart -DUSIM -o srad_IDP

clean:
	rm -f srad_IDP *.txt gpgpusim_power_report* _cuobjdump_*
