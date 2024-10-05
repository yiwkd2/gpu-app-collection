#!/usr/bin/make

default:
	nvcc -g -Wno-deprecated-gpu-targets add_vectors.cu -lcudart -o add_vectors_IDP

clean:
	rm -rf add_vectors_IDP gpgpusim_power_report* _cuobjdump_* *.txt
