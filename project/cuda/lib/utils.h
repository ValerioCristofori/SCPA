#include <time.h>


struct matrix{
	int 			*I;  //row index
	int 			*J;  //column index
	double 			*val;
	int 			nz;
	int 			 M;
	int 			 N;
};


struct vector{
	int 			xdim; 
	double 			*X;
};



struct Csr{
	int    *IRP;
	int   	*JA;
	double 	*AS;
	int 	  M;
	int 	  N;
};


struct Ellpack{
	int   	*JA_t;
	double 	*AS_t;
	int 	  M;
	int 	  N;
	int     *MAXNZ;
};

struct Result{
	double  *res;
	int 		 len;
	long     elapsed_time;
	double 	 gpuflops;
};




typedef struct{
	time_t tv_sec; /* seconds */
	long tv_nsec; /* nanoseconds */
}timespect;


int checkerror(const double *resp, const double *ress, int dim);

void getmul(struct matrix *mat, struct vector *vec, double* res);

void quicksort(double* a, double* vindex, int* rindex, int* cindex, int n);

void dprintArrayInt(int* a, int len);

void dprintArrayDouble(double* a, int len);




int load_matrix(char *matrix_filename, struct matrix* mat);

int load_vector(char *vector_filename, struct vector* vec, int M);

int calculate_prod(struct matrix *mat, struct vector* vec, double *res_seq, char* mode, FILE *fpt);

