#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include "mmio.h"
#include "utils.h"


int main(int argc, char *argv[])
{
    int ret_code;
    MM_typecode matcode;
    FILE *f;
    int M, N, nz, xdim;  
    int i, j, *I, *J, *IRP;
    double *val, *x, *res, *vIndex;

    if (argc < 3)
	{
		fprintf(stderr, "Usage: %s [martix-market-filename] [vector-filename]\n", argv[0]);
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

    if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) !=0)
        exit(1);


    /* reseve memory for matrices */

    I = (int *) malloc(nz * sizeof(int));  // row index
    J = (int *) malloc(nz * sizeof(int));  // column index
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
    if (xdim >= N)
    {
        xdim = N;
    } else {
        printf("dimension vector too small!\n");
        exit(1);
    }
    x = (double*)malloc(xdim * sizeof(double));
    for (i = 0; i<xdim; i++)
    {
        fscanf(f, "%lg\n", &x[i]);
    }

    if (f != stdin) fclose(f);


    /************************/
    /* now write out matrix */
    /************************/

    //mm_write_banner(stdout, matcode);
    //mm_write_mtx_crd_size(stdout, M, N, nz);
    //for (i=0; i<nz; i++)
    //    fprintf(stdout, "%d %d %20.19g\n", I[i]+1, J[i]+1, val[i]);


    /* preprocessing the matrix */
    //the original calculation result
    double* res_seq = (double*)malloc(M*sizeof(double));
    memset(res_seq, 0, M*sizeof(double));

    getmul(val, x, I, J, nz, res_seq);

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

    dprintArrayInt(IRP, M+1);

    /* Start serial computation and start timer */


    printf("\n Start computation ... \n");
    struct timeval start, end;

    res = (double*)malloc(M*sizeof(double));
    memset(res, 0, M*sizeof(double));
    
    gettimeofday(&start, NULL);
    for (i = 0; i<M; i++)
    {
        for (j = IRP[i]; j <= IRP[i+1] - 1; j++)
        {
          int tmp = J[j];
            res[i] += val[j] * x[tmp];
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

