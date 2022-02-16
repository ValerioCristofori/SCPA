#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h> 
#include "utils.h"

#define FILENAME "omp-metrics.csv"



int             num_threads;
FILE            *fpt;



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

    
    if (argc < 4)
    {
        fprintf(stderr, "Usage: %s [-serial/-ompCSR/-ompELLPACK/-cudaCSR/-cudaELLPACK] [martix-market-filename] [vector-filename] {num threads IF openMP}\nFor the third argument:\n", argv[0]);
        exit(1);
    }

    struct matrix* mat = (struct matrix*) malloc(sizeof(struct matrix));
    struct vector* vec = (struct vector*) malloc(sizeof(struct vector));

    if( !strcmp(argv[1],"-serial")){

    }else if(!strcmp(argv[1],"-ompCSR") ){
        if( argc == 4 ){
            fprintf(stderr, "Error in openMP calculation: specify number of threads\n");
            exit(1);
        }
        num_threads = atoi(argv[4]);
        omp_set_num_threads(num_threads);

    }else if(!strcmp(argv[1],"-ompELLPACK") ){
        if( argc == 4 ){
            fprintf(stderr, "Error in openMP calculation: specify number of threads\n");
            exit(1);
        }
        num_threads = atoi(argv[4]);
        omp_set_num_threads(num_threads);

    }else if(!strcmp(argv[1],"-cudaCSR") ){

    }else if(!strcmp(argv[1],"-cudaELLPACK") ){

    }else{
        fprintf(stderr, "Usage: %s [-serial/-ompCSR/-ompELLPACK/-cudaCSR/-cudaELLPACK] [martix-market-filename] [vector-filename] {num threads IF openMP}\nFor the third argument:\n", argv[0]);
        exit(1);
    }
    ret = load_mat_vec(argv[2], argv[3], mat, vec);
    if( ret == -1 )
        exit(1);

    /* calculate the product result sequentially for testing */
    double *res_seq = (double*)malloc((mat->M)*sizeof(double));
    memset(res_seq, 0, (mat->M)*sizeof(double));

    getmul(mat, vec, res_seq);

    ret = calculate_prod(mat, vec, res_seq, argv[1], num_threads);
    if( ret == -1 )
        exit(1);


    free(mat);
    free(vec);

    


}