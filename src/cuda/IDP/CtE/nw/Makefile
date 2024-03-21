#!/usr/bin/make

# CUDA compiler
NVCC = nvcc
NVCC_FLAGS = -g -Wno-deprecated-gpu-targets

default:
	$(NVCC) $(NVCC_FLAGS) ${KERNEL_DIM} needle.cu -lcudart -o nw_CtE

clean:
	rm -f nw_CtE *.txt gpgpusim_power_report* _cuobjdump_*
