#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "lib/utils.h"


struct Csr* preprocess_csr(struct matrix *mat)
{
    struct Csr *csr_mat;

    int M = mat->M;
    int N = mat->N;
    int nz = mat->nz;
    int *I = mat->I;
    int *J = mat->J;
    double *val = mat->val;

    //preprocess the dataset to make the calculation can be parallelized
    double *vIndex = (double*)malloc(nz*sizeof(double));
    memset(vIndex, 0, nz*sizeof(double));
    for (int i = 0; i < nz; i++)
    {
        vIndex[i] = (double)I[i] * N + J[i];
        if (vIndex[i] < 0)
        {   
               printf("Error: %lg < 0\n", vIndex[i]);
               return NULL;
            }
    }

    quicksort(val, vIndex, I, J, nz);

    int *IRP = (int*)malloc((M+1)*sizeof(int)); //start position of each row
    memset(IRP, -1, (M+1)*sizeof(int));


    for (int i = 0; i<nz; i++)
    {
        int tmp = (int)(vIndex[i] / N);
        if (IRP[tmp] == -1)
        {
            IRP[tmp] = i;
        }

    }
    // update last entry in IRP array with the greater one
    IRP[M] = nz;

    csr_mat = (struct Csr*) malloc(sizeof(struct Csr));
    csr_mat->M = M;
    csr_mat->N = N;
    csr_mat->JA = J;
    csr_mat->AS = val;
    csr_mat->IRP = IRP;

    free(vIndex);

    return csr_mat;
}



struct Ellpack* preprocess_ellpack(struct matrix *mat)
{

    struct Ellpack *ellpack_mat;

    int maxnz = 0;
    int M = mat->M;
    int N = mat->N;
    int nz = mat->nz;
    int *I = mat->I;
    int *J = mat->J;
    double *val = mat->val;

    //preprocess the dataset to make the calculation can be parallelized
    double *vIndex = (double*)malloc(nz*sizeof(double));
    memset(vIndex, 0, nz*sizeof(double));
    for (int i = 0; i < nz; i++)
    {
        vIndex[i] = (double)I[i] * N + J[i];
        if (vIndex[i] < 0)
        {   
               printf("Error: %lg < 0\n", vIndex[i]);
               return NULL;
            }
    }

    quicksort(val, vIndex, I, J, nz);

    free(vIndex);
    
    int count_nz = 1;
    for (int i = 0; i<nz-1; i++)
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
    int *JA = (int *) malloc((maxnz * M) * sizeof(int));
    memset(JA, 0, (maxnz*M)*sizeof(int));
    double *AS = (double *) malloc((maxnz * M) * sizeof(double));
    memset(AS, 0, (maxnz*M)*sizeof(double));

    // populate the 2D arrays
    int x = 0, y = 0;
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
        y = J[h];
        JA[x*maxnz + count - 1] = y;
        AS[x*maxnz + count - 1] = val[h];

    }
    // lastline
    while( count < maxnz ){
        JA[x*maxnz + count] = y;
        count++;
    }

    ellpack_mat = (struct Ellpack*) malloc(sizeof(struct Ellpack));
    ellpack_mat->M = M;
    ellpack_mat->N = N;
    ellpack_mat->maxnz = maxnz;
    ellpack_mat->JA = JA;
    ellpack_mat->AS = AS;

    return ellpack_mat;
}



struct Result* module_serial_csr(struct Csr* csr_mat, struct vector* vec)
{
    int         M = csr_mat->M;
    int        *IRP = csr_mat->IRP;
    int        *JA = csr_mat->JA;
    double     *AS = csr_mat->AS;
    double     *X = vec->X;

    printf("\n Start computation ... \n");
    struct timeval start, end;



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

    long elapsed_time = ((end.tv_sec * 1000000 + end.tv_usec)
          - (start.tv_sec * 1000000 + start.tv_usec));

    printf(" Total time: %ld micro-seconds\n\n",  elapsed_time);

    struct Result* res_serial_csr = (struct Result*) malloc(sizeof(struct Result));
    res_serial_csr->res = res;
    res_serial_csr->len = M;
    res_serial_csr->elapsed_time = elapsed_time;
    

    return res_serial_csr;
}




struct Result* module_omp_csr(struct Csr* csr_mat, struct vector* vec, int thread_num)
{
    int j, ckey;

    int *IRP = csr_mat->IRP;
    int *JA = csr_mat->JA;
    double *AS = csr_mat->AS;
    double *X = vec->X;
    int M = csr_mat->M;


    printf("\n Start computation ... \n");
    struct timeval start, end;

    double *res = (double*)malloc(M*sizeof(double));
    memset(res, 0, M*sizeof(double));
    
    int chunk = 1000;

    gettimeofday(&start, NULL);
    #pragma omp parallel num_threads(thread_num)
    {
    #pragma omp for private( j, ckey ) schedule( dynamic, chunk )
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

    long elapsed_time = ((end.tv_sec * 1000000 + end.tv_usec)
          - (start.tv_sec * 1000000 + start.tv_usec));


    printf(" Total time: %ld micro-seconds\n\n",  elapsed_time);

    struct Result* res_omp_csr = (struct Result*) malloc(sizeof(struct Result));
    res_omp_csr->res = res;
    res_omp_csr->len = M;
    res_omp_csr->elapsed_time = elapsed_time;

    return res_omp_csr;
}




struct Result* module_omp_ellpack(struct Ellpack* ellpack_mat, struct vector* vec, int thread_num)
{
    int ja;

    double *X = vec->X;
    double *AS = ellpack_mat->AS;
    int   *JA = ellpack_mat->JA;
    int maxnz = ellpack_mat->maxnz;
    int M = ellpack_mat->M;

    printf("\n Start computation ... \n");
    struct timeval start, end;

    double *res = (double*)malloc(M*sizeof(double));
    memset(res, 0, M*sizeof(double));

    int chunk = 1000;

    gettimeofday(&start, NULL);
    int j;
    #pragma omp parallel num_threads(thread_num)
    {
    #pragma omp for private( j, ja ) schedule( dynamic, chunk )
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

    long elapsed_time = ((end.tv_sec * 1000000 + end.tv_usec)
          - (start.tv_sec * 1000000 + start.tv_usec));

    printf(" Total time: %ld micro-seconds\n\n",  elapsed_time);

    struct Result* res_omp_ellpack = (struct Result*) malloc(sizeof(struct Result));
    res_omp_ellpack->res = res;
    res_omp_ellpack->len = M;
    res_omp_ellpack->elapsed_time = elapsed_time;

    return res_omp_ellpack;
}


int calculate_prod(struct matrix *mat, struct vector* vec, double *res_seq, char* mode, int num_threads, FILE *fpt)
{
    double *res;
    int     len;
    long elapsed_time;
    int     passed = 0;
    
    

    if( !strcmp(mode,"-serial")){

            struct Csr *csr_mat;

            printf("Start Serial calculation with CSR...\n");

            csr_mat = preprocess_csr(mat);
            if( csr_mat == NULL )
                return 1;
            /* serial calculation with csr */
            struct Result *res_serial_csr = module_serial_csr(csr_mat, vec);

            res = res_serial_csr->res;
            len = res_serial_csr->len;
            elapsed_time = res_serial_csr->elapsed_time;

            free(csr_mat);
            free(res_serial_csr);



    }else if(!strcmp(mode,"-ompCSR") ){

            struct Csr *csr_mat;
        
            printf("Start OpenMP calculation with CSR...\n");

            csr_mat = preprocess_csr(mat);
            if( csr_mat == NULL )
                return 1;
            /* calculation with csr with OpenMP */
            struct Result *res_omp_csr = module_omp_csr(csr_mat, vec, num_threads);

            res = res_omp_csr->res;
            len = res_omp_csr->len;
            elapsed_time = res_omp_csr->elapsed_time;

            free(csr_mat);

    }else if(!strcmp(mode,"-ompELLPACK") ){
        
            struct Ellpack *ellpack_mat;

            printf("Start OpenMP calculation with ELLPACK...\n");

            /* preprocess and build ellpack format */
            ellpack_mat = preprocess_ellpack(mat);
            if( ellpack_mat == NULL )
                return 1;
            /* calculation with ellpack with OpenMP */
            struct Result *res_omp_ellpack = module_omp_ellpack(ellpack_mat, vec, num_threads);

            res = res_omp_ellpack->res;
            len = res_omp_ellpack->len;
            elapsed_time = res_omp_ellpack->elapsed_time;
            

            free(ellpack_mat);


    }

    if (!checkerror(res, res_seq, len))
    {
        printf("Calculation Error!\n");

    }
    else {
        printf(" Test Result Passed ... \n");
        passed = 1;
    }

    /* print on file the entry result */
    fprintf(fpt,"%s, %ld, 0.0, %d\n", mode, elapsed_time, passed);
    fflush(fpt);


    free(res);
    free(res_seq);    
    

    return 0;
}