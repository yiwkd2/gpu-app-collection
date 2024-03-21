#!/usr/bin/make

default:
	nvcc -g -Wno-deprecated-gpu-targets sssp.cu -lcudart -o sssp_CtE

clean:
	rm -rf sssp_CtE gpgpusim_power_report* _cuobjdump_* *.txt
