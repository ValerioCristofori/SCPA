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
    int M, N, nz, xdim;  
    int i, j, *I, *J, *JA, ja;
    int  x = 0, y = 0, maxnz = 0;
    double value = 0.0;
    double *val, *X, *res, *vIndex, *AS;

    if (argc < 4)
	{
		fprintf(stderr, "Usage: %s [martix-market-filename] [vector-filename] [num-threads]\n", argv[0]);
		exit(1);
	}
    else    
    { 
        if ((f = fopen(argv[1], "r")) == NULL) 
            exit(1);
    }

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        exit(1);
    }


    /*  This is how one can screen matrix types if their application */
    /*  only supports a subset of the Matrix Market data types.      */

    if (mm_is_complex(matcode) && mm_is_matrix(matcode) && 
            mm_is_sparse(matcode) )
    {
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        exit(1);
    }

    /* find out size of sparse matrix */

    if ( mm_read_mtx_crd_size(f, &M, &N, &nz) !=0)
        exit(1);


    /* reseve memory for matrices */

    I = (int *) malloc(nz * sizeof(int));
    J = (int *) malloc(nz * sizeof(int));
    val = (double *) malloc(nz * sizeof(double));


    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (i=0; i<nz; i++)
    {
        fscanf(f, "%d %d %lg\n", &I[i], &J[i], &val[i]);
        I[i]--;  /* adjust from 1-based to 0-based */
        J[i]--;
    }

    if (f !=stdin) fclose(f);


    /* Open and load the vector input for the product */  
    if ((f = fopen(argv[2], "r")) == NULL)
    {
        printf("Fail to open the input vector file!\n");
        exit(1);
    }
    fscanf(f, "%d\n", &xdim);
    if (xdim > M)
    {
        xdim = M;
    } else {
        printf("dimension vector too small!\n");
        exit(1);
    }
    X = (double*)malloc(xdim * sizeof(double));
    for (i = 0; i<xdim; i++)
    {
        fscanf(f, "%lg\n", &X[i]);
    }

    if (f != stdin) fclose(f);


    /* preprocessing the matrix */
    //the original calculation result
    double* res_seq = (double*)malloc(M*sizeof(double));
    memset(res_seq, 0, M*sizeof(double));

    getmul(val, X, I, J, nz, res_seq);

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
    
    int count_nz = 1;
    for (i = 0; i<nz-1; i++)
    {
        if ( I[i] == I[i+1] ){
            count_nz++;
        }else{
            if( count_nz > maxnz )
                maxnz = count_nz;
            count_nz = 1;
        }

    }
    if( count_nz > maxnz )
        maxnz = count_nz;
    if ( maxnz > nz )
    {
        printf("MaxNZ Error!\n");
        exit(1);
    }
    
    // reserve JA and AS 2D arrays
    JA = (int *) malloc((maxnz * M) * sizeof(int));
    memset(JA, -1, (maxnz*M)*sizeof(int));
    AS = (double *) malloc((maxnz * M) * sizeof(double));
    memset(AS, 0, (maxnz*M)*sizeof(double));

    // populate the 2D arrays
    int prev = 0;
    int count = 0;
    for ( int h = 0; h < nz; h++ )
    {
        x = I[h];

        if( prev == x ){
            count++;
        }else{
            // fill the rest of row with the latest value
            for( int k = 0; k < maxnz - count; k++ ){
                JA[prev*maxnz + count + k] = y;
            }
            count = 0;

            prev = x;
            h--;
            continue;
        }

        prev = x;
        y = J[h];
        value = val[h];
        JA[x*maxnz + count - 1] = y;
        AS[x*maxnz + count - 1] = value;

    }
    // lastline
    while( count < maxnz ){
        JA[x*maxnz + count] = y;
        count++;
    }


    // set num thread for openMP
    int thread_num = atoi(argv[3]);
    //omp_set_num_threads(thread_num); 


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
    #pragma omp for private( j, ja ) schedule( dynamic, chunk )
    for( i=0; i<M; i++ ){
        double result = 0.0;
        for( j = 0; j < maxnz; j++ ){
            
            ja = JA[i*maxnz + j];
                result += AS[i*maxnz + j] * X[ja];
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
    free(X);
    free(I);
    free(J);
    free(val);
}
