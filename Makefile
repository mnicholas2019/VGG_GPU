.PHONY: all
all: clean main

main: 
	nvcc -I ../NVIDIA_CUDA-10.1_Samples/common/inc/ -o main main.cu

.Phony: run
run: clean main
	./main

.Phony: prof
prof: run
	nvprof ./main

.Phony: clean
clean:
	rm -f main


