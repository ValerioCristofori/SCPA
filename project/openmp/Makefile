# compiler flags:
#  -g    adds debugging information to the executable file
#  -Wall turns on most, but not all, compiler warnings
CFLAGS  = -g -Wall
OMP     = -fopenmp

serial:	serial-CSR
openmp-CSR: omp-CSR
openmp-ELLPACK: omp-ELLPACK


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



#Utils
mmio.o:   mmio.c   mmio.h
	gcc -c mmio.c  $(CFLAGS)

utils.o:	utils.c utils.h
	gcc -c utils.c $(CFLAGS)

clean:
	rm -f serial-CSR omp-CSR omp-ELLPACK output.txt *.o
