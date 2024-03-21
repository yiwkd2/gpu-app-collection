#!/usr/bin/make

default:
	nvcc -g -Wno-deprecated-gpu-targets sssp.cu -lcudart -DUSIM -o sssp_IDP

clean:
	rm -rf sssp_IDP gpgpusim_power_report* _cuobjdump_* *.txt
