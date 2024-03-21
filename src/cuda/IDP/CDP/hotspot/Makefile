#!/usr/bin/make

# CUDA compiler
NVCC = nvcc
NVCC_FLAGS = -g -Wno-deprecated-gpu-targets

SRC = hotspot.cu

default:
	$(NVCC) $(NVCC_FLAGS) $(KERNEL_DIM) hotspot.cu -lcudart -o hotspot_CDP_hw
	$(NVCC) $(NVCC_FLAGS) $(KERNEL_DIM) hotspot.cu -lcudart -DUSIM -o hotspot_CDP

clean:
	rm -f hotspot_CDP_hw hotspot_CDP *.txt gpgpusim_power_report* _cuobjdump_*
