#!/usr/bin/make

default:
	nvcc -g -Wno-deprecated-gpu-targets 2DConvolution.cu -lcudart -o 2dconv_ZC

clean:
	rm -rf 2dconv_ZC gpgpusim_power_report* _cuobjdump_* *.txt
