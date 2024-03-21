#!/usr/bin/make

# CUDA compiler
NVCC = nvcc
NVCC_FLAGS = -g -Wno-deprecated-gpu-targets

default:
	$(NVCC) $(NVCC_FLAGS) $(KERNEL_DIM) backprop.cu -lcudart -DUSIM -o backprop_IDP

clean: $(SRC)
	rm -f backprop_IDP *.txt gpgpusim_power_report* _cuobjdump_*
