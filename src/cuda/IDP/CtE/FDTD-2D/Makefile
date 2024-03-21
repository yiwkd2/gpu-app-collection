#!/usr/bin/make

default:
	nvcc -g -Wno-deprecated-gpu-targets fdtd2d.cu -lcudart -o fdtd2d_CtE

clean:
	rm -rf fdtd2d_CtE gpgpusim_power_report* _cuobjdump_* *.txt
