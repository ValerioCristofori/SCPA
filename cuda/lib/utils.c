#include <stdio.h>
#include <stdlib.h>

#include "mmio.h"
#include "utils.h"

#define max(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })


/* To classify if a calculation has been successful, 
    look at the rel diff and the diff. 
    If the rel diff is less than the rounding of the unit 
    (which according to the IEEE standard is 1.19e-07). */
#define ERROR_BOUND     1.19e-07



int load_matrix(char *matrix_filename, struct matrix *mat)
{
    MM_typecode  matcode;
    FILE              *f;

    int         nz, M, N;
    int           *I_tmp, *J_tmp;
    double          *val_tmp;

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
    I_tmp = (int *) malloc(nz * sizeof(int));
    J_tmp = (int *) malloc(nz * sizeof(int));
    val_tmp = (double *) malloc(nz * sizeof(double));

    if( mm_is_pattern(matcode) ){
        for (int i=0; i<nz; i++)
        {
            fscanf(f, "%d %d\n", &I_tmp[i], &J_tmp[i]);
            val_tmp[i] = 1.0;
            I_tmp[i]--;  /* adjust from 1-based to 0-based */
            J_tmp[i]--;  
        }

    }else{
        for (int i=0; i<nz; i++)
        {
            fscanf(f, "%d %d %lg\n", &I_tmp[i], &J_tmp[i], &val_tmp[i]);
            I_tmp[i]--;  /* adjust from 1-based to 0-based */
            J_tmp[i]--;
            
        }
    }       

    int *I, *J;
    double *val;

    if( mm_is_symmetric(matcode) ){
       
        //count real non-zeros
        int real_nz = nz;
        for(int i=0; i<nz; i++){
            if( I_tmp[i] != J_tmp[i] ){
                real_nz += 1;
            }
        }
        printf("real_nz: %d\n", real_nz);

        I = (int *) malloc(real_nz * sizeof(int));
        J = (int *) malloc(real_nz * sizeof(int));
        val = (double *) malloc(real_nz * sizeof(double));

        int nz_in_diag = 0;
        for(int i=0; i<nz; i++){
            I[i] = I_tmp[i];
            J[i] = J_tmp[i];
            val[i] = val_tmp[i];

            if( I_tmp[i] != J_tmp[i] ){
                I[nz + i - nz_in_diag] = J_tmp[i];
                J[nz + i - nz_in_diag] = I_tmp[i];
                val[nz + i - nz_in_diag] = val_tmp[i];
            }else{
                nz_in_diag++;
            }
        }

        nz = real_nz;

    }else{
        I = I_tmp;
        J = J_tmp;
        val = val_tmp;
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

int load_vector(char *vector_filename, struct vector *vec, int N)
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
    if (xdim > N)
    {
        xdim = N;
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

// calculate the relative difference and the absolute max difference
    // between res[i] and res_seq[i] -> use for testing if the calculation is successful done
int checkerror(struct Result* result, const double* res_seq, int dim_res_seq)
{
    int i;
    int j = 0;
    double *res = result->res;
    int dim = result->len;
    double maxabs;
    double reldiff = 0.0;
    double diff    = 0.0;
    printf("Dim: %d dim_seq: %d\n", dim, dim_res_seq );

    // check if relative diff is less than the rounding of the unit
    if( dim != dim_res_seq ){
        for (i = 0; i < dim_res_seq; i++)
        {
            
            if(res_seq[i] == 0){
                //empty row
                if( res[j] != 0){
                    continue;
                }
            }

            printf("Res_seq: %lg  Res: %lg\n", res_seq[i], res[j] );

            maxabs = max( abs(res_seq[i]), abs(res[j]));
            if (maxabs == 0.0) maxabs=1.0;
            reldiff = max(reldiff, abs(res_seq[i] - res[j])/maxabs);
            diff = max(diff, abs(res_seq[i] - res[j]));

            j++;
        }
        if( j != dim ){
            printf("Dim: %d j: %d\n", dim, j );
            result->reldiff = reldiff;
            result->diff    = diff;
            return 0;
        }
    }else{

        for (i = 0; i < dim; i++)
        {
            printf("Res_seq: %lg  Res: %lg\n", res_seq[i], res[i] );
            maxabs = max( abs(res_seq[i]), abs(res[i]));
            if (maxabs == 0.0) maxabs=1.0;
            reldiff = max(reldiff, abs(res_seq[i] - res[i])/maxabs);
            diff = max(diff, abs(res_seq[i] - res[i]));

        }

    }
    result->reldiff = reldiff;
    result->diff    = diff;
    if( reldiff <= ERROR_BOUND )
        return 1;

    return 0;

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