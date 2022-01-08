#include <time.h>

typedef struct{
	time_t tv_sec; /* seconds */
	long tv_nsec; /* nanoseconds */
}timespect;

int checkerror(const double *resp, const double *ress, int dim);

void getmul(const double *val, const double *vec, const int *rIndex,
	const int *cIndex, int nz, double *res);

void quicksort(double* a, double* vindex, int* rindex, int* cindex, int n);

void dprintArrayInt(int* a, int len);

void dprintArrayDouble(double* a, int len);