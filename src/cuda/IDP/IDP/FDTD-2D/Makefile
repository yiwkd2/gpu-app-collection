#!/usr/bin/make

default:
	nvcc -g -Wno-deprecated-gpu-targets fdtd2d.cu -lcudart -DUSIM -o fdtd2d_IDP

clean:
	rm -rf fdtd2d_IDP gpgpusim_power_report* _cuobjdump_* *.txt
