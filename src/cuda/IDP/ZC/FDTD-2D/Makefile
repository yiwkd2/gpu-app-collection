#!/usr/bin/make

default:
	nvcc -g -Wno-deprecated-gpu-targets fdtd2d.cu -lcudart -o fdtd2d_ZC

clean:
	rm -rf fdtd2d_ZC gpgpusim_power_report* _cuobjdump_* *.txt
