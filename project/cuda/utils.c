#include "utils.h"
#include <stdio.h>


void getmul(const float* val, const float* vec, const int* rIndex, const int* cIndex, int nz, float* res)
{
	int i; 
	for (i = 0; i < nz; i++)
	{
		int rInd = rIndex[i];
		int cInd = cIndex[i];
		res[rInd] += val[i] * vec[cInd];
	}
}

int checkerror(const float* resp, const float* ress, int dim)
{
	int i;
	for (i = 0; i < dim; i++)
	{
		if (resp[i] != ress[i])
			return 0;
	}

	return 1;

}

void quicksort(float* a, float* vindex, int* rindex, int* cindex, int n)
{
	int i, j, m;

	float p, t, s;
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

void dprintArrayDouble(float* a, int len){
	int i;
	for (i=0; i<len; i++)
        fprintf(stdout, "%lg\t", a[i]);
    printf("\n");
}