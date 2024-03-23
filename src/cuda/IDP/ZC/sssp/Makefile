#!/usr/bin/make

default:
	nvcc -g -Wno-deprecated-gpu-targets sssp.cu -lcudart -o sssp_ZC

clean:
	rm -rf sssp_ZC gpgpusim_power_report* _cuobjdump_* *.txt
