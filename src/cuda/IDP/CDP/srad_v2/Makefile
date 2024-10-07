#!/usr/bin/make

# CUDA compiler
NVCC = nvcc
NVCC_FLAGS = -g -Wno-deprecated-gpu-targets

default:
	$(NVCC) $(NVCC_FLAGS) $(KERNEL_DIM) srad.cu -lcudart -o srad_CDP_hw
	$(NVCC) $(NVCC_FLAGS) $(KERNEL_DIM) srad.cu -lcudart -DPREFETCH -o srad_CDP_prefetch_hw
	$(NVCC) $(NVCC_FLAGS) $(KERNEL_DIM) srad.cu -lcudart -DUSIM -o srad_CDP

clean:
	rm -f srad_CDP_hw srad_CDP_prefetch_hw srad_CDP *.txt gpgpusim_power_report* _cuobjdump_*
