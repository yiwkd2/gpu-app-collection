#!/usr/bin/make

default:
	nvcc -g -Wno-deprecated-gpu-targets 2DConvolution.cu -lcudart -o 2dconv_CDP_hw
	nvcc -g -Wno-deprecated-gpu-targets 2DConvolution.cu -lcudart -DPREFETCH -o 2dconv_CDP_prefetch_hw
	nvcc -g -Wno-deprecated-gpu-targets 2DConvolution.cu -lcudart -DUSIM -o 2dconv_CDP

clean:
	rm -rf 2dconv_CDP_hw 2dconv_CDP_prefetch_hw 2dconv_CDP gpgpusim_power_report* _cuobjdump_* *.txt
