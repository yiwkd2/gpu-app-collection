#!/usr/bin/make

# CUDA compiler
NVCC = nvcc
NVCC_FLAGS = -g -Wno-deprecated-gpu-targets

default:
	$(NVCC) $(NVCC_FLAGS) $(KERNEL_DIM) backprop.cu -lcudart -o backprop_CDP_hw
	$(NVCC) $(NVCC_FLAGS) $(KERNEL_DIM) backprop.cu -lcudart -DUSIM -o backprop_CDP

clean: $(SRC)
	rm -f backprop_CDP_hw backprop_CDP *.txt gpgpusim_power_report* _cuobjdump_*
