#include <stdio.h>
#include <stdlib.h>

#include "mmio.h"
#include "utils.h"



int load_matrix(char *matrix_filename, struct matrix *mat)
{
    MM_typecode matcode;
    FILE             *f;

	int        nz, M, N;
	int          *I, *J;
	double         *val;

	if ((f = fopen(matrix_filename, "r")) == NULL) 
        return -1;

	if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        return -1;
    }

    if (mm_is_complex(matcode) && mm_is_matrix(matcode) && 
            mm_is_sparse(matcode) )
    {
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        return -1;
    }

    /* find size of sparse matrix */
    if ( mm_read_mtx_crd_size(f, &M, &N, &nz) !=0)
        return -1;


    /* allocate memory for matrix */
    I = (int *) malloc(nz * sizeof(int));
    J = (int *) malloc(nz * sizeof(int));
    val = (double *) malloc(nz * sizeof(double));


    for (int i=0; i<nz; i++)
    {
        fscanf(f, "%d %d %lg\n", &I[i], &J[i], &val[i]);
        I[i]--;  /* adjust from 1-based to 0-based */
        J[i]--;
    }

    if (f !=stdin) fclose(f);


    
    mat->I = I;
    mat->J = J;
    mat->val = val;
    mat->nz = nz;
    mat->M = M;
    mat->N = N;

    return 0;

}


int load_vector(char *vector_filename, struct vector *vec, int M)
{

    FILE        *f;
	int 	  xdim;
	double 		*X;

	/* load the vector input for the product */  
    if ((f = fopen(vector_filename, "r")) == NULL)
    {
        printf("Fail to open the input vector file!\n");
        return -1;
    }
    fscanf(f, "%d\n", &xdim);
    if (xdim > M)
    {
        xdim = M;
    } else {
        printf("dimension vector too small!\n");
        return -1;
    }
    X = (double*)malloc(xdim * sizeof(double));
    for (int i = 0; i<xdim; i++)
    {
        fscanf(f, "%lg\n", &X[i]);
    }

    if (f != stdin) fclose(f);

    vec->X = X;
    vec->xdim = xdim;
    
    return 0;
}


/* sequential product calculation */
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

/* order input vectors that have the same length n
    computation time -> O(n*log(n))  */
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