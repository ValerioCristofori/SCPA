# compiler flags:
#  -g    adds debugging information to the executable file
#  -Wall turns on most, but not all, compiler warnings
CFLAGS  = -g -Wall
OMP     = -fopenmp
OBJS    = mmio.o utils.o calc.o preproc.o
LIB 	= lib


#Main
main:	main.o	$(OBJS)
	gcc -o main *.o $(CFLAGS) $(OMP)

main.o:	main.c $(LIB)/utils.h
	gcc -c main.c $(CFLAGS) $(OMP)

#Preproc
preproc.o: $(LIB)/preprocessing.c $(LIB)/utils.h
	gcc -c $(LIB)/preprocessing.c  $(CFLAGS)

#Calc
calc.o:	calc.c $(LIB)/utils.h
	gcc -c calc.c $(CFLAGS) $(OMP)


#Utils
mmio.o:   $(LIB)/mmio.c   $(LIB)/mmio.h
	gcc -c $(LIB)/mmio.c  $(CFLAGS)

utils.o:	$(LIB)/utils.c $(LIB)/utils.h
	gcc -c $(LIB)/utils.c $(CFLAGS)

clean:
	rm -f main *.o
