#!/usr/bin/make

default:
	nvcc -g -Wno-deprecated-gpu-targets 2DConvolution.cu -lcudart -DUSIM -o 2dconv_IDP

clean:
	rm -rf 2dconv_IDP gpgpusim_power_report* _cuobjdump_* *.txt
