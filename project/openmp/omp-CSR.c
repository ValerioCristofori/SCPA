#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h> 
#include <sys/time.h>
#include "mmio.h"
#include "utils.h"


int main(int argc, char *argv[])
{
    MM_typecode matcode;
    FILE *f;
    int M, N, nz, xdim, ckey;  
    int i, j, *I, *J, *IRP;
    double *val, *x, *res, *vIndex;










    //preprocess the dataset to make the calculation can be parallelized
    vIndex = (double*)malloc(nz*sizeof(double));
    memset(vIndex, 0, nz*sizeof(double));
    for (i = 0; i < nz; i++)
    {
        vIndex[i] = (double)I[i] * N + J[i];
        if (vIndex[i] < 0)
        {   
               printf("Error!\n");
               exit(1);
            }
    }

    quicksort(val, vIndex, I, J, nz);

    IRP = (int*)malloc((M+1)*sizeof(int)); //start position of each row
    memset(IRP, -1, (M+1)*sizeof(int));

    for (i = 0; i<nz; i++)
    {
        int tmp = (int)(vIndex[i] / N);
        if (IRP[tmp] == -1)
        {
            IRP[tmp] = i;
        }

    }
    // update last entry in IRP array with the greater one
    IRP[M] = nz + 1;

    // set num thread for openMP
    int thread_num = atoi(argv[3]);
    omp_set_num_threads(thread_num); 


    /* Start parallel computation with OpenMP */
    /*    and start timer                     */

    printf("\n Start computation ... \n");
    struct timeval start, end;

    res = (double*)malloc(M*sizeof(double));
    memset(res, 0, M*sizeof(double));
    
    int chunk = 1000;

    gettimeofday(&start, NULL);
    #pragma omp parallel num_threads(thread_num)
    {
    #pragma omp for private( j, ckey ) schedule( dynamic, chunk )
    for( i=0; i<M; i++ ){
        double result = 0.0;
        for( j = IRP[i]; j <= IRP[i+1]-1; j++ ){
            
            ckey = J[j];
                result += val[j] * x[ckey];
        }
        res[i] = result;
    }

    }
    gettimeofday(&end, NULL);


    /* End computation and timer */

    printf(" End of computation ... \n \n");

    long elapsed_time = ((end.tv_sec * 1000000 + end.tv_usec)
          - (start.tv_sec * 1000000 + start.tv_usec));
  
    
    if (!checkerror(res, res_seq, M))
    {
        printf("Calculation Error!\n");
        exit(1);
    }
    else {
        printf(" Test Result Passed ... \n");
    }


    printf(" Total time: %ld micro-seconds\n\n",  elapsed_time);



    /* Print out the result in a file -> output.txt */

    if ((f = fopen("output.txt", "w")) == NULL)
    {
        printf("Fail to open the output file!\n");
        exit(1);
    }
    for (i = 0; i<xdim; i++)
    {
        fprintf(f, "%lg\n", res[i]);
    }
    fclose(f);


    /* Free all dynamic variables */

    free(res_seq);
    free(vIndex);
    free(res);
    free(x);
    free(I);
    free(J);
    free(val);
    free(IRP);
}

