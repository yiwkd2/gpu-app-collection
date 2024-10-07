#!/usr/bin/make

default:
	nvcc -g -Wno-deprecated-gpu-targets sssp.cu -lcudart -o sssp_CDP_hw
	nvcc -g -Wno-deprecated-gpu-targets sssp.cu -lcudart -DPREFETCH -o sssp_CDP_prefetch_hw
	nvcc -g -Wno-deprecated-gpu-targets sssp.cu -lcudart -DUSIM -o sssp_CDP

clean:
	rm -rf sssp_CDP_hw sssp_CDP_prefetch_hw sssp_CDP gpgpusim_power_report* _cuobjdump_* *.txt
