# compiler flags:
#  -g    adds debugging information to the executable file
#  -Wall turns on most, but not all, compiler warnings
CFLAGS  = -g -Wall
OMP     = -fopenmp

serial:	serial-CSR
openmp-CSR: omp-CSR
openmp-ELLPACK: omp-ELLPACK
cuda-CSR: cuda-CSR
cuda-ELLPACK: cuda-ELLPACK


#Serial
serial-CSR:	serial-CSR.o mmio.o utils.o
	gcc -o serial-CSR *.o $(CFLAGS)

serial-CSR.o:	serial-CSR.c
	gcc -c serial-CSR.c $(CFLAGS)


#OpenMP
omp-CSR:	omp-CSR.o	mmio.o utils.o
	gcc -o omp-CSR *.o $(CFLAGS) $(OMP)

omp-CSR.o:	omp-CSR.c
	gcc -c omp-CSR.c $(CFLAGS) $(OMP)


omp-ELLPACK: omp-ELLPACK.o mmio.o utils.o
	gcc -o omp-ELLPACK *.o $(CFLAGS) $(OMP)

omp-ELLPACK.o:	omp-ELLPACK.c
	gcc -c omp-ELLPACK.c $(CFLAGS) $(OMP)



#CUDA
cuda-CSR:	cuda-CSR.o mmio.o utils.o
	nvcc -i cuda-CSR *.o $(CFLAGS)

cuda-CSR.o:	cuda-CSR.c kernel-CSR.c
	nvcc -c cuda-CSR.c kernel-CSR.c $(CFLAGS)


cuda-ELLPACK: cuda-ELLPACK.o mmio.o utils.o
	nvcc -o cuda-ELLPACK *.o $(CFLAGS)

cuda-ELLPACK.o:	cuda-ELLPACK.c kernel-ELLPACK.c
	nvcc -c cuda-ELLPACK.c kernel-ELLPACK.c $(CFLAGS)



#Utils
mmio.o:   mmio.c   mmio.h
	gcc -c mmio.c  $(CFLAGS)

utils.o:	utils.c utils.h
	gcc -c utils.c $(CFLAGS)

clean:
	rm -f serial-CSR omp-CSR omp-ELLPACK cuda-CSR cuda-ELLPACK output.txt *.o
