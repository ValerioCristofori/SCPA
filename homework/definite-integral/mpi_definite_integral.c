#include <stdio.h>
#include "mpi.h"

double f(double x) { return 2*x+1; } // polinomio facile

int main(int argc, char *argv[])
{
 	int myid, numprocs,volba=0;
    unsigned long int i, n;
    double startwtime, endwtime, a = 4.0, b = 5.0;
    double total, integral = 0.0;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);  // obtain number
                                               // of processes
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);  // obtain process number

     do {
		 if (myid == 0) {
			printf("\n Enter number of intervals (0 to exit): "); fflush(stdout);
			scanf("%ld", &n);
			if(n==0) break;
			startwtime = MPI_Wtime();
		 }
		 // send variable n, with vector length 1, type int, from process #0 to others
		 MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD); 

		 if (n != 0) {
			double h = (b - a) / n;
			double i1 = myid*(n/ numprocs);
			double i2 = (myid+1)*(n/numprocs);
			integral= ( f(a+i1*h) + f(a+i2*h) ) / 2;
			for( i=i1+1 ; i<i2 ; i++ )
			integral += f(a+i*h);
		}
		// summarize integral variable from all processes to total in process 0,
		// length 1, type double, operation sum/addition
		MPI_Reduce(&integral, &total, 1, MPI_DOUBLE, MPI_SUM, 0, 
		MPI_COMM_WORLD);

		if (myid == 0) {
		  endwtime = MPI_Wtime();
		  printf("I= %f\n", total);
		  printf("spent time: %f s\n", endwtime - startwtime); fflush(stdout);
		}
	  } while (n != 0);
	  MPI_Finalize();
	  return 0;
}