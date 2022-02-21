#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "utils.h"

struct Csr* preprocess_csr(struct matrix *mat)
{
    struct Csr *csr_mat;

    int M   = mat->M;
    int N   = mat->N;
    int nz  = mat->nz;
    int *I  = mat->I;
    int *J  = mat->J;
    double *val = mat->val;



    //preprocess the dataset to make the calculation can be parallelized
    /* build vector of Index that specify a vector 1D of the corresponding matrix 2D
        of the non-zeros positions */ 
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
    // order values vector through vIndex
    quicksort(val, vIndex, I, J, nz);

    int empty_rows = 0;
    int gap;
    for( int i = 0; i < nz - 1; i++){
        gap = I[i+1] - I[i];
        if( gap > 1 ){
            // empty row
            empty_rows += gap - 1;
        }
    }
    int *tmp_IRP = (int*)malloc((M+1)*sizeof(int)); //start position of each row
    memset(tmp_IRP, -1, (M+1)*sizeof(int));

    /* build IRP vector */
    for (int i = 0; i<nz; i++)
    {
        int tmp = (int)(vIndex[i] / N);
        if (tmp_IRP[tmp] == -1)
        {
            tmp_IRP[tmp] = i;
        }

    }
    // update last entry in IRP array with the greater one
    tmp_IRP[M] = nz;

    printf("Empty rows: %d\n", empty_rows);
    int *IRP;
    if(empty_rows != 0){
        IRP = (int*)malloc((M - empty_rows + 1)*sizeof(int)); //start position of each row
        memset(IRP, -1, (M - empty_rows + 1)*sizeof(int));
        int index = 0;
        for(int i = 0; i < M + 1; i++){
            if( tmp_IRP[i] != -1 ){
                IRP[index] = tmp_IRP[i];
                index++;
            }
        }
        M -= empty_rows;

    }else{
        IRP = tmp_IRP;
    }


    csr_mat = (struct Csr*) malloc(sizeof(struct Csr));
    csr_mat->M = M;
    csr_mat->N = N;
    csr_mat->nz = nz;
    csr_mat->JA = J;
    csr_mat->AS = val;
    csr_mat->IRP = IRP;

    free(vIndex);

    return csr_mat;
}



struct Ellpack* preprocess_ellpack(struct matrix *mat)
{

    struct Ellpack *ellpack_mat;

    int *JA_t;
    double *AS_t;

    int maxnz = 0;
    int M = mat->M;
    int N = mat->N;
    int nz = mat->nz;
    int *I = mat->I;
    int *J = mat->J;
    double *val = mat->val;

    //preprocess the dataset to make the calculation can be parallelized
    /* build vector of Index that specify a vector 1D of the corresponding matrix 2D
        of the non-zeros positions */
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
    // order values vector through vIndex
    quicksort(val, vIndex, I, J, nz);

    free(vIndex);
    
    // count maxnz in the current matrix 
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

    // reserve JA and AS and MAXNZ arrays
    int *JA = (int *) malloc((maxnz * M) * sizeof(int));
    memset(JA, 0, (maxnz*M)*sizeof(int));
    double *AS = (double *) malloc((maxnz * M) * sizeof(double));
    memset(AS, 0, (maxnz*M)*sizeof(double));
    int *MAXNZ = (int *) malloc( M* sizeof(int));
    memset(MAXNZ, -1, M*sizeof(int));


    // populate the arrays
    int x = 0, y = 0;    // temp values: x -> row idx, y -> col idx
    int prev        = 0; // last row index value
    int count       = 0; // idx for update JA and AS
    int count_row   = 0; // num of nz in current row
    for ( int h = 0; h < nz; h++ )
    {
        x = I[h];

        if( prev == x ){
            count++;
        }else{
            //new row
            MAXNZ[count_row] = count; // set number of nz in last row
            count_row++;

            count = 0;
            prev = x;
            h--;
            continue;
        }

        prev = x;
        y = J[h];
        JA[x*maxnz + count - 1] = y;
        AS[x*maxnz + count - 1] = val[h];

    }
    MAXNZ[count_row] = count;

    //transposition vector AS e JA
    //reserve memory
    JA_t = (int *) malloc((maxnz*M) * sizeof(int));
    memset(JA_t, -1, (maxnz*M)*sizeof(int));
    AS_t = (double *) malloc((maxnz*M) * sizeof(double));
    memset(AS_t, 0, (maxnz*M)*sizeof(double));

    // transposition of JA and AS to JA_t and AS_t
    for(int i=0; i<M; i++)
    {
        for(int j=0; j<maxnz; j++)
        {
            JA_t[j*M+i]=JA[i*maxnz+j];
            AS_t[j*M+i]=AS[i*maxnz+j];
        }
    }

    ellpack_mat = (struct Ellpack*) malloc(sizeof(struct Ellpack));
    ellpack_mat->M = M;
    ellpack_mat->N = N;
    ellpack_mat->nz = nz;
    ellpack_mat->maxnz = maxnz;
    ellpack_mat->MAXNZ = MAXNZ;
    ellpack_mat->JA_t = JA_t;
    ellpack_mat->AS_t = AS_t;

    return ellpack_mat;
}
