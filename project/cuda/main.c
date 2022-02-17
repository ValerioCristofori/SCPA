#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lib/utils.h"

#define FILENAME "test-metrics.csv"


int load_mat_vec(char* mat_filename, char* vector_filename, struct matrix* mat, struct vector* vec){

    int M, ret;

    /* preprocess matrix: from .mtx to matrix */
    ret = load_matrix(mat_filename, mat);
    if( ret == -1 )
        return -1;
    M = mat->M;

    /* load vector */
    ret = load_vector(vector_filename, vec, M);
    if( ret == -1 )
        return -1;

    return 0;
}



int main(int argc, char *argv[])
{
    
    int ret;
    FILE *fpt;

    
    if (argc < 4)
    {
        fprintf(stderr, "Usage: %s [-cudaCSR/-cudaELLPACK] [martix-market-filename] [vector-filename]\n", argv[0]);
        exit(1);
    }

    //check if output file exists -> create
    if ( (fpt = fopen(FILENAME, "r")) == NULL) 
    {
        fpt = fopen(FILENAME, "w+");
        fprintf(fpt,"Matrix, M, N, nz, CalculationMode, CalculationTime, GPUFlops, Passed\n");
        fflush(fpt);
    }
    fpt = fopen(FILENAME, "a");



    struct matrix* mat = (struct matrix*) malloc(sizeof(struct matrix));
    struct vector* vec = (struct vector*) malloc(sizeof(struct vector));

    if(!strcmp(argv[1],"-cudaCSR") ){

    }else if(!strcmp(argv[1],"-cudaELLPACK") ){

    }else{
        fprintf(stderr, "Usage: %s [-cudaCSR/-cudaELLPACK] [martix-market-filename] [vector-filename]\n", argv[0]);
        goto exit;
    }
    ret = load_mat_vec(argv[2], argv[3], mat, vec);
    if( ret == -1 )
        goto exit;

    /* calculate the product result sequentially for testing */
    double *res_seq = (double*)malloc((mat->M)*sizeof(double));
    memset(res_seq, 0, (mat->M)*sizeof(double));

    getmul(mat, vec, res_seq);

    // write on file
    fprintf(fpt,"%s, %d, %d, %d, ", strrchr(argv[2], '/'), mat->M, mat->N, mat->nz);

    ret = calculate_prod(mat, vec, res_seq, argv[1], fpt);
    if( ret == -1 )
        goto exit;

exit:
    fclose(fpt);
    free(mat);
    free(vec);

    


}