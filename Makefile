CFLAGS=$(OPT) --std=c++11 -g -O3
MODULE  := class1 conv1

.PHONY: all clean

all: $(MODULE)

INCLUDES  = -I ../NVIDIA_CUDA-10.1_Samples/common/inc

class1: classifiers.cu
	nvcc $^ $(CFLAGS) $(INCLUDES) -o $@ -DNi=25088 -DNn=4096 -DTii=512 -DTi=64 -DTnn=32  -DTn=16
	

conv1: convolutions.cu
	nvcc $^ $(CFLAGS) $(INCLUDES) -o $@ -DNx=224 -DNy=224 -DKx=3  -DKy=3 -DNi=64 -DNn=64 -DTii=32 -DTi=16 -DTnn=32 -DTn=16 -DTx=7 -DTy=7


clean:
	@rm -f $(MODULE)
	rm -f convd
