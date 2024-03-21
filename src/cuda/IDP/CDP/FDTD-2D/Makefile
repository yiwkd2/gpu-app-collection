#!/usr/bin/make

default:
	nvcc -g -Wno-deprecated-gpu-targets fdtd2d.cu -lcudart -o fdtd2d_CDP_hw
	nvcc -g -Wno-deprecated-gpu-targets fdtd2d.cu -lcudart -DUSIM -o fdtd2d_CDP

clean:
	rm -rf fdtd2d_CDP_hw fdtd2d_CDP gpgpusim_power_report* _cuobjdump_* *.txt
