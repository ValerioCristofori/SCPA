# compiler flags:
#  -g    adds debugging information to the executable file
#  -Wall turns on most, but not all, compiler warnings
CFLAGS  = -g -Wall

serial:	serial-CSR
openMP: omp-CSR

#Serial
serial-CSR:	serial-CSR.o mmio.o utils.o
	gcc -o serial-CSR *.o $(CFLAGS)

serial-CSR.o:	serial-CSR.c
	gcc -c serial-CSR.c $(CFLAGS)
#OpenMP
omp-CSR:	omp-CSR.o	mmio.o utils.o
	gcc -o omp-CSR *.o $(CFLAGS)

omp-CSR.o:	omp-CSR.c
	gcc -c omp-CSR.c $(CFLAGS)

#Utils
mmio.o:   mmio.c   mmio.h
	gcc -c mmio.c  $(CFLAGS)

utils.o:	utils.c utils.h
	gcc -c utils.c $(CFLAGS)

clean:
	rm -f omp-CSR serial-CSR output.txt *.o
