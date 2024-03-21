#!/usr/bin/make

# CUDA compiler
NVCC = nvcc
NVCC_FLAGS = -g -Wno-deprecated-gpu-targets

default:
	$(NVCC) $(NVCC_FLAGS) ${KERNEL_DIM} needle.cu -lcudart -o nw_CDP_hw
	$(NVCC) $(NVCC_FLAGS) ${KERNEL_DIM} needle.cu -lcudart -DUSIM -o nw_CDP

clean:
	rm -f nw_CDP_hw nw_CDP *.txt gpgpusim_power_report* _cuobjdump_*
