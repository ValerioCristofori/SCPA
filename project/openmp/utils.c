#include "utils.h"
#include <stdio.h>

#include "mmio.h"



struct matrix* load_matrix(char *matrix_filename)
{
    MM_typecode matcode;
    FILE *f;
	
    struct *matrix;
	int nz, M, N;
	int *I, *J;
	double *val;

	if ((f = fopen(matrix_filename, "r")) == NULL) 
        exit(1);

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


    matrix = (struct *matrix) malloc(sizeof(struct matrix))
    matrix->I = I;
    matrix->J = J;
    matrix->val = val;
    matrix->nz = nz;
    matrix->M = M;
    matrix->N = N;

    return matrix;

}


struct vector* load_vector(char *vector_filename){

    struct 	 *vector;
	int 		xdim;
	double 		  *X;

	    /* Open and load the vector input for the product */  
    if ((f = fopen(vector_filename, "r")) == NULL)
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

    vector = (struct *vector) malloc(sizeof(struct vector))
    vector->X = X;
    vector->xdim = xdim;


    return vector;
}



void getmul(struct matrix *mat, struct vector *vec, double* res)
{
	int i; 
	for (i = 0; i < mat->nz; i++)
	{
		int rInd = mat->I[i];
		int cInd = mat->J[i];
		res[rInd] += mat->val[i] * vec->X[cInd];
	}
}

bool checkerror(const double* resp, const double* ress, int dim)
{
	int i;
	for (i = 0; i < dim; i++)
	{
		if (resp[i] != ress[i])
			return false;
	}

	return true;

}

void quicksort(double* a, double* vindex, int* rindex, int* cindex, int n)
{
	int i, j, m;

	double p, t, s;
	if (n < 2)
		return;
	p = vindex[n / 2];

	for (i = 0, j = n - 1;; i++, j--) {
		while (vindex[i]<p)
			i++;
		while (p<vindex[j])
			j--;
		if (i >= j)
			break;
		t = a[i];
		a[i] = a[j];
		a[j] = t;

		s = vindex[i];
		vindex[i] = vindex[j];
		vindex[j] = s;

		m = rindex[i];
		rindex[i] = rindex[j];
		rindex[j] = m;

		m = cindex[i];
		cindex[i] = cindex[j];
		cindex[j] = m;
	}
	quicksort(a, vindex, rindex, cindex, i);
	quicksort(a + i, vindex + i, rindex + i, cindex + i, n - i);
}


void dprintArrayInt(int* a, int len){
	int i;
	for (i=0; i<len; i++)
        fprintf(stdout, "%d\t", a[i]);
    printf("\n");
}

void dprintArrayDouble(double* a, int len){
	int i;
	for (i=0; i<len; i++)
        fprintf(stdout, "%lg\t", a[i]);
    printf("\n");
}