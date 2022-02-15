#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h> 
#include <sys/time.h>
#include "mmio.h"
#include "utils.h"



int main(int argc, char *argv[])
{
	struct matrix 	*mat;
	struct vector 	*vec;
	double		*res_seq;

	int M, N;

	if (argc < 4)
	{
		fprintf(stderr, "Usage: %s [martix-market-filename] [vector-filename] [0/1/2] {num threads IF openMP}\n
			For the third argument:\n\t0: Serial calculation\n\t1: calculation with OpenMP\n\t2: calculation with CUDA\n", argv[0]);
		exit(1);
	}
    else    
    { 
        if (argc == 4 && argv[3] == 1){
        	fprintf(stderr, "Error in openMP calculation: specify number of threads\n");
			exit(1);
        }
    }

    /* preprocess matrix: from .mtx to matrix */
    mat = load_matrix(argv[1]);
    M = mat->M;
    N = mat->N;

    /* load vector */
    vec = load_vector(argv[2]);

    /* calculate the product result sequentially for testing */
    res_seq = (double*)malloc(M*sizeof(double));
    memset(res_seq, 0, M*sizeof(double));

    getmul(mat, vec, res_seq);



    switch(argv[3]){
    	case 0:

    	case 1:

    	case 2:


    	default:
    		exit(1);
    }

}