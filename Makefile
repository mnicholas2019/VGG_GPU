.PHONY: all
all: main

main: 
	nvcc -I ../NVIDIA_CUDA-10.1_Samples/common/inc/ -o main main.cu

.Phony: clean
clean:
	rm -f main
