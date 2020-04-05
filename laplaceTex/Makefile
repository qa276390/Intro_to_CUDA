CUDIR     := /usr/local/nvidia

CC      := gcc
CFLAGS  := -O3 -Wall -g
CPP     := g++
CFLAGS  := -O3 -Wall -g
NVCC    := $(CUDIR)/bin/nvcc
NVFLAGS := -O3 -I$(CUDIR)/include -m64 -arch=compute_52 -code=sm_52 -Xptxas -v

BIN     := laplace
BIN_TEX := laplaceTex
BIN_CPU := laplace_cpu
BIN_TEX_HW3_2 := hw3_2_laplaceTex
LIB     :=

all: $(BIN) $(BIN_TEX) $(BIN_CPU) $(BIN_TEX_HW3_2)

$(BIN): laplace.cu
	$(NVCC) -o $(BIN) $(NVFLAGS) laplace.cu $(LIB)

$(BIN_TEX): laplaceTex.cu
	$(NVCC) -o $(BIN_TEX) $(NVFLAGS) laplaceTex.cu $(LIB)

$(BIN_TEX_HW3_2): hw3_2_laplaceTex.cu
	$(NVCC) -o $(BIN_TEX_HW3_2) $(NVFLAGS) hw3_2_laplaceTex.cu $(LIB)

$(BIN_CPU): laplace_cpu.cu
	$(NVCC) -o $(BIN_CPU) $(NVFLAGS) laplace_cpu.cu $(LIB)
clean:
	rm -f $(BIN) $(BIN_CPU) $(BIN_TEX) $(BIN_TEX_HW3_2)
