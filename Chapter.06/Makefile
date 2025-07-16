# Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
NP ?= 4
NVCC=nvcc
CUDA_HOME ?= /usr/local/cuda
ifndef MPI_HOME
$(error MPI_HOME is not set)
endif

GENCODE_SM70    := -gencode arch=compute_70,code=sm_70
GENCODE_SM80    := -gencode arch=compute_80,code=sm_80 -gencode arch=compute_80,code=compute_80
GENCODE_FLAGS	:= $(GENCODE_SM70) $(GENCODE_SM80)


NVCC_FLAGS_OPTIMIZED = -Xptxas --optimize-float-atomics
NVCC_FLAGS_CUB = -DHAVE_CUB
NVCC_FLAGS_DEFAULT =


NVCC_FLAGS += -dc -Xcompiler -fopenmp -lineinfo $(GENCODE_FLAGS) -std=c++14  -I$(MPI_HOME)/include
NVCC_LDFLAGS = -ccbin=mpic++ -L$(NVSHMEM_HOME)  -L$(MPI_HOME)/lib -lmpi -L$(CUDA_HOME)/lib64 -lcuda -lcudart 


optimized: CURRENT_NVCC_FLAGS = $(NVCC_FLAGS_OPTIMIZED)
optimized: jacobi

cub: CURRENT_NVCC_FLAGS = $(NVCC_FLAGS_OPTIMIZED) $(NVCC_FLAGS_CUB)
cub: jacobi

default: CURRENT_NVCC_FLAGS =  $(NVCC_FLAGS_DEFAULT)
default: jacobi


jacobi: Makefile jacobi.cu
	$(NVCC) $(CURRENT_NVCC_FLAGS) $(NVCC_FLAGS) jacobi.cu -c -o jacobi.o
	$(NVCC) $(GENCODE_FLAGS) jacobi.o -o jacobi $(NVCC_LDFLAGS)

clean:
	rm -f jacobi jacobi.o *.nsys-rep jacobi.*.compute-sanitizer.log