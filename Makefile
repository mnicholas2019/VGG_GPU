CFLAGS=$(OPT) --std=c++11 -g -O3
MODULE  := class1

.PHONY: all clean

all: $(MODULE)

INCLUDES  = -I ../NVIDIA_CUDA-10.1_Samples/common/inc

class1: classifiers.cu
	nvcc $^ $(CFLAGS) $(INCLUDES) -o $@ -DNi=25088 -DNn=4096 -DTii=512 -DTi=64 -DTnn=32  -DTn=16
	#nvcc $^ $(CFLAGS) $(INCLUDES) -o $@ -DNi=4 -DNn=2 -DTii=512 -DTi=64 -DTnn=32  -DTn=16

clean:
	@rm -f $(MODULE)
