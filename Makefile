CFLAGS=$(OPT) --std=c++11 -g -O3
MODULE  := class1 class2 conv1 conv2

.PHONY: all clean

all: $(MODULE)

INCLUDES  = -I ../NVIDIA_CUDA-10.1_Samples/common/inc

class1: classifiers.cu
	nvcc $^ $(CFLAGS) $(INCLUDES) -o $@ -DNi=25088 -DNn=4096 -DTii=512 -DTi=64 -DTnn=32  -DTn=16
	
class2: classifiers.cu
	nvcc $^ $(CFLAGS) $(INCLUDES) -o $@ -DNi=4096 -DNn=1024 -DTii=32 -DTi=32 -DTnn=32  -DTn=16

conv1: convolutions.cu
	nvcc $^ $(CFLAGS) $(INCLUDES) -o $@ -DNx=224 -DNy=224 -DKx=3 -DKy=3 -DNi=64 -DNn=64 -DTii=32 -DTi=16 -DTnn=32 -DTn=16 -DTx=7 -DTy=7

conv2: convolutions.cu
	nvcc $^ $(CFLAGS) $(INCLUDES) -o $@ -DNx=14 -DNy=14 -DKx=3 -DKy=3 -DNi=512 -DNn=512 -DTii=32 -DTi=16  -DTnn=32 -DTn=16 -DTx=2 -DTy=2

clean:
	@rm -f $(MODULE)

