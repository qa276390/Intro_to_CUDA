CUDIR   := /usr/local/nvidia

CC      := g++
CFLAGS  := -O2 -Wall
NVCC    := nvcc
NVFLAGS := -O3 -I$(CUDIR)/include -m64 -arch=compute_52 -code=sm_52 -Xptxas -v -rdc=true
LIB     := -lcufft -lcudart

BIN1    := fft_cpu
BIN2    := fft_cuFFT
BIN3    := poisson_cuFFT
BIN4    := _poisson_cuFFT

all: $(BIN1) $(BIN2) $(BIN3) $(BIN4)

$(BIN1): fft_cpu.cpp
	$(CC) -o $(BIN1) $(CFLAGS) $<

$(BIN2): fft_cuFFT.cu
	$(NVCC) -o $(BIN2) $(NVFLAGS) $< $(LIB)

$(BIN3): poisson_cuFFT.cu
	$(NVCC) -o $(BIN3) $(NVFLAGS) $< $(LIB)

$(BIN4): _poisson_cuFFT.cu
	$(NVCC) -o $(BIN4) $(NVFLAGS) $< $(LIB)

clean:
	rm -f $(BIN1) $(BIN2) $(BIN3) $(BIN4) *.o
