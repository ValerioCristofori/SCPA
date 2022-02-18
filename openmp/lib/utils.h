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
	int   	*JA;
	double 	*AS;
	int 	  M;
	int 	  N;
	int   maxnz;
};

struct Result{
	double  *res;
	int 		 len;
	double    elapsed_time;
};




typedef struct{
	time_t tv_sec; /* seconds */
	double tv_nsec; /* nanoseconds */
}timespect;

typedef enum{
  true = 1, false = 0
}bool;


bool checkerror(const double *resp, const double *ress, int dim);

void getmul(struct matrix *mat, struct vector *vec, double* res);

void quicksort(double* a, double* vindex, int* rindex, int* cindex, int n);

void dprintArrayInt(int* a, int len);

void dprintArrayDouble(double* a, int len);



struct Csr* preprocess_csr(struct matrix *mat);

struct Ellpack* preprocess_ellpack(struct matrix *mat);




int load_matrix(char *matrix_filename, struct matrix* mat);

int load_vector(char *vector_filename, struct vector* vec, int M);

int calculate_prod(struct matrix *mat, struct vector* vec, double *res_seq, char* mode, int num_threads, FILE *fpt);

