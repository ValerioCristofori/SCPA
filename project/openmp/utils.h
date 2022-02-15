#include <time.h>

typedef struct{
	int 			*I;  //row index
	int 			*J;  //column index
	double 	*val;
	int 			nz;
	int 			 M;
	int 			 N;
}matrix;


typedef struct{
	int 			xdim; 
	double 			*X;
}vector;


typedef struct{
	time_t tv_sec; /* seconds */
	long tv_nsec; /* nanoseconds */
}timespect;

typedef enum{
  true = 1, false = 0
}bool;


struct matrix* load_matrix(char *matrix_filename);

struct vector* load_vector(char *vector_filename);



bool checkerror(const double *resp, const double *ress, int dim);

void getmul(struct matrix *mat, struct vector *vec, double* res);

void quicksort(double* a, double* vindex, int* rindex, int* cindex, int n);

void dprintArrayInt(int* a, int len);

void dprintArrayDouble(double* a, int len);