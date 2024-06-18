#!/usr/bin/make

default:
	nvcc -g -Wno-deprecated-gpu-targets add_vectors.cu -lcudart -o add_vectors_MS

clean:
	rm -rf add_vectors_MS gpgpusim_power_report* _cuobjdump_* *.txt
