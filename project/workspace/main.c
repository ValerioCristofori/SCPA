#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h> 
#include "utils.h"

#ifdef TEST
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#define FILENAME "omp-metrics.csv"
#endif


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




#ifdef TEST
int test()
{
    struct matrix   *mat;
    struct vector   *vec;
    double          *res_seq;

    struct dirent *dp;
    DIR *dfd;

    int ret;

    char *matrix_dir = "../tests/matrix";
    char *vector_filename = "../tests/vector/vector.txt";

    fpt = fopen(FILENAME, "w"); 
    if (fpt == NULL){
        fprintf(stderr, "Can't open %s\n", FILENAME);
        return -1;
    }
    fprintf(fpt,"Matrix, Serial-CSR, OpenMP-CSR, OpenMP-ELLPACK, Passed\n");

    printf("Starting test...\n");

    /* iterate over matrix files */

     if ((dfd = opendir(matrix_dir)) == NULL)
     {
        fprintf(stderr, "Can't open %s\n", matrix_dir);
        return -1;
     }

     char mat_filename[257] ;

     while ((dp = readdir(dfd)) != NULL)
     {
          struct stat stbuf ;
          sprintf( mat_filename , "%s/%s",matrix_dir,dp->d_name);
          if( stat(mat_filename,&stbuf ) == -1 )
          {
               printf("Unable to stat file: %s\n",mat_filename);
               continue ;
          }

          if ( ( stbuf.st_mode & S_IFMT ) == S_IFDIR )
          {
               continue;
               // Skip directories
          }
          else
          {
                printf("File: %s\n",mat_filename);
                fprintf( fpt, "%s, ", strrchr(mat_filename, '/'));

                mat = (struct matrix*) malloc(sizeof(struct matrix));
                vec = (struct vector*) malloc(sizeof(struct vector));

                ret = load_mat_vec(mat_filename, vector_filename, mat, vec);
                if( ret == -1 )
                    return -1;

                /* calculate the product result sequentially for testing */
                res_seq = (double*)malloc((mat->M)*sizeof(double));
                memset(res_seq, 0, (mat->M)*sizeof(double));

                getmul(mat, vec, res_seq);

                ret = calculate_prod_test(mat, vec, res_seq, num_threads, fpt);

                if( ret == -1 )
                    return -1;


                free(mat);
                free(vec);
                free(res_seq);

                mat = NULL;
                vec = NULL;
                res_seq = NULL;
          }
     }
     return 0;
}
#endif


int main(int argc, char *argv[])
{
    
    int ret;


    #ifdef TEST
    printf("Test Mode\n");
    
    if( argc > 1 )
    {
       
        /* starting test mode */
        if( argc == 1 || atoi(argv[1]) == 0){
            fprintf(stderr, "Error in openMP calculation: specify number of threads\n");
            fprintf(stderr, "Usage: %s [num threads]\n", argv[0]);
            exit(1);
        }
        if(argc >= 1 && atoi(argv[1]) != 0){
            num_threads = atoi(argv[1]);
            omp_set_num_threads(num_threads);
        }

        ret = test();
        if( ret == -1 )
            exit(1);
        return 1;
        
    }else{
        fprintf(stderr, "Error in openMP calculation: specify number of threads\n");
        fprintf(stderr, "Usage: %s [num threads]\n", argv[0]);
        exit(1);
    }
    
    #else
    
    if (argc < 4)
    {
        fprintf(stderr, "Usage: %s [martix-market-filename] [vector-filename] [0/1] {num threads IF openMP}\nFor the third argument:\n\t0: Serial calculation\n\t1: calculation with OpenMP\n", argv[0]);
        exit(1);
    }
    else    
    { 
        if (argc == 4 && atoi(argv[3]) == 1){
            fprintf(stderr, "Error in openMP calculation: specify number of threads\n");
            exit(1);
        }
        if( argc > 4 && atoi(argv[3]) == 1){
            num_threads = atoi(argv[4]);
            omp_set_num_threads(num_threads);
        }

    }

    struct matrix* mat = (struct matrix*) malloc(sizeof(struct matrix));
    struct vector* vec = (struct vector*) malloc(sizeof(struct vector));

    ret = load_mat_vec( argv[1], argv[2], mat, vec);
    if( ret == -1 )
        exit(1);

    /* calculate the product result sequentially for testing */
    double *res_seq = (double*)malloc((mat->M)*sizeof(double));
    memset(res_seq, 0, (mat->M)*sizeof(double));

    getmul(mat, vec, res_seq);

    ret = calculate_prod(mat, vec, res_seq, atoi(argv[3]), num_threads);
    if( ret == -1 )
        exit(1);

    free(mat->I);
    free(mat->J);
    free(mat);
    free(vec->X);
    free(vec);
    free(res_seq);
    #endif


    

	
}