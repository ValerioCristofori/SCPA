#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <sys/time.h>

#include "lib/utils.h"

/* openmp divides the iteration into chunk and distributes
    Each thread executes a chunk of iterations
and then requests another chunk until there are no more chunks available. */
#define CHUNK_SIZE 1000


struct Result* module_serial_csr(struct Csr* csr_mat, struct vector* vec)
{
    int           M = csr_mat->M;
    int          nz = csr_mat->nz;
    int        *IRP = csr_mat->IRP;
    int         *JA = csr_mat->JA;
    double      *AS = csr_mat->AS;
    double       *X = vec->X;

    printf("\n Start computation ... \n");
    double flopcnt=2.e-6*nz;
    struct timeval start, end;

    /* allocate memory for result vector */
    double *res = (double*)malloc(M*sizeof(double));
    memset(res, 0, M*sizeof(double));

    gettimeofday(&start, NULL);
    for (int i = 0; i<M; i++)
    {
        for (int j = IRP[i]; j <= IRP[i+1] - 1; j++)
        {
          int tmp = JA[j];
            res[i] += AS[j] * X[tmp];
        }
    }
    gettimeofday(&end, NULL);


    /* End computation and timer */

    printf(" End of computation ... \n \n");

    long time = ((end.tv_sec * 1000000 + end.tv_usec)
          - (start.tv_sec * 1000000 + start.tv_usec));
    double elapsed_time = (double)time / 1000;
    double cpuflops = flopcnt / elapsed_time;

    printf(" Total time: %lg milliseconds\n\n",  elapsed_time);

    struct Result* res_serial_csr = (struct Result*) malloc(sizeof(struct Result));
    res_serial_csr->res = res;
    res_serial_csr->len = M;
    res_serial_csr->elapsed_time = elapsed_time;
    res_serial_csr->cpuflops = cpuflops;
    

    return res_serial_csr;
}




struct Result* module_omp_csr(struct Csr* csr_mat, struct vector* vec, int thread_num)
{
    int j, ckey;

    int     *IRP = csr_mat->IRP;
    int      *JA = csr_mat->JA;
    double   *AS = csr_mat->AS;
    double    *X = vec->X;
    int        M = csr_mat->M;
    int       nz = csr_mat->nz;


    printf("\n Start computation ... \n");
    double flopcnt=2.e-6*nz;
    struct timeval start, end;

    /* allocate memory for result vector */
    double *res = (double*)malloc(M*sizeof(double));
    memset(res, 0, M*sizeof(double));
    

    gettimeofday(&start, NULL);
    #pragma omp parallel num_threads(thread_num)
    {
    #pragma omp for private( j, ckey ) schedule( dynamic, CHUNK_SIZE )
    for( int i=0; i<M; i++ ){
        double result = 0.0;
        for( j = IRP[i]; j <= IRP[i+1]-1; j++ ){
            
            ckey = JA[j];
                result += AS[j] * X[ckey];
        }
        res[i] = result;
    }

    }
    gettimeofday(&end, NULL);


    /* End computation and timer */
    printf(" End of computation ... \n \n");

    long time = ((end.tv_sec * 1000000 + end.tv_usec)
          - (start.tv_sec * 1000000 + start.tv_usec));
    double elapsed_time = (double)time / 1000;
    double cpuflops = flopcnt / elapsed_time;

    printf(" Total time: %lg milliseconds\n\n",  elapsed_time);

    struct Result* res_omp_csr = (struct Result*) malloc(sizeof(struct Result));
    res_omp_csr->res = res;
    res_omp_csr->len = M;
    res_omp_csr->elapsed_time = elapsed_time;
    res_omp_csr->cpuflops = cpuflops;

    return res_omp_csr;
}




struct Result* module_omp_ellpack(struct Ellpack* ellpack_mat, struct vector* vec, int thread_num)
{
    int ja;

    double      *X = vec->X;
    double     *AS = ellpack_mat->AS;
    int        *JA = ellpack_mat->JA;
    int      maxnz = ellpack_mat->maxnz;
    int          M = ellpack_mat->M;
    int         nz = ellpack_mat->nz;

    printf("\n Start computation ... \n");
    double flopcnt=2.e-6*nz;
    struct timeval start, end;

    /* allocate memory for result vector */
    double *res = (double*)malloc(M*sizeof(double));
    memset(res, 0, M*sizeof(double));


    gettimeofday(&start, NULL);
    int j;
    #pragma omp parallel num_threads(thread_num)
    {
    #pragma omp for private( j, ja ) schedule( dynamic, CHUNK_SIZE )
    for( int i=0; i<M; i++ ){
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

    long time = ((end.tv_sec * 1000000 + end.tv_usec)
          - (start.tv_sec * 1000000 + start.tv_usec));
    double elapsed_time = (double)time / 1000;
    double cpuflops = flopcnt / elapsed_time;

    printf(" Total time: %lg milliseconds\n\n",  elapsed_time);

    struct Result* res_omp_ellpack = (struct Result*) malloc(sizeof(struct Result));
    res_omp_ellpack->res = res;
    res_omp_ellpack->len = M;
    res_omp_ellpack->elapsed_time = elapsed_time;
    res_omp_ellpack->cpuflops = cpuflops;

    return res_omp_ellpack;
}



int calculate_prod(struct matrix *mat, struct vector* vec, double *res_seq, char* mode, int num_threads, FILE *fpt)
{
    double      *res;         // result of the parallel product
    int         len;          // len of the result
    double      elapsed_time; // time spent in the calculation
    double      cpuflops;     // CPU floating point ops per second
    int         passed = 0;   // 1 if the parallelized product is successful 
    struct Result   *result;
    
    /* select the right matrix preprocessing and calculation mode
      with respect to the 'mode' value entered   */
    if( !strcmp(mode,"-serial")){

            struct Csr *csr_mat;

            printf("Start Serial calculation with CSR...\n");

            /* pre-processing the matrix following CSR format */
            csr_mat = preprocess_csr(mat);
            if( csr_mat == NULL )
                return -1;
            
            /* serial calculation with csr */
            result = module_serial_csr(csr_mat, vec);

            free(csr_mat);


    }else if(!strcmp(mode,"-ompCSR") ){

            struct Csr *csr_mat;
        
            printf("Start OpenMP calculation with CSR with %d Threads...\n", num_threads);

            /* pre-processing the matrix following CSR format */
            csr_mat = preprocess_csr(mat);
            if( csr_mat == NULL )
                return -1;
            
            /* calculation with csr with OpenMP */
            result = module_omp_csr(csr_mat, vec, num_threads);

            free(csr_mat);

    }else if(!strcmp(mode,"-ompELLPACK") ){
        
            struct Ellpack *ellpack_mat;

            printf("Start OpenMP calculation with ELLPACK with %d Threads...\n", num_threads);

            /* preprocess and build ellpack format for matrix */
            ellpack_mat = preprocess_ellpack(mat);
            if( ellpack_mat == NULL )
                return -1;
            
            /* calculation with ellpack with OpenMP */
            result = module_omp_ellpack(ellpack_mat, vec, num_threads);

            free(ellpack_mat);


    }

    // build vars for verification and metrics
    res = result->res;
    len = result->len;
    elapsed_time = result->elapsed_time;
    cpuflops = result->cpuflops;

    // check if parallel calculation is successful done
    if (!checkerror(res, res_seq, len))
    {
        printf("Calculation Error!\n");

    }
    else {
        printf(" Test Result Passed ... \n");
        passed = 1;
    }

    /* print on file the entry result */
    fprintf(fpt,"%s,%d,%lg,%lg,%d\n", mode, num_threads, elapsed_time, cpuflops, passed);
    fflush(fpt);


    free(result->res);
    free(result);
    free(res_seq);    
    

    return 0;
}