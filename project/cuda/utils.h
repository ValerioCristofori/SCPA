#include <time.h>

typedef struct{
	time_t tv_sec; /* seconds */
	long tv_nsec; /* nanoseconds */
}timespect;

int checkerror(const float *resp, const float *ress, int dim);

void getmul(const float *val, const float *vec, const int *rIndex,
	const int *cIndex, int nz, float *res);

void quicksort(float* a, float* vindex, int* rindex, int* cindex, int n);

void dprintArrayInt(int* a, int len);

void dprintArrayFloat(float* a, int len);