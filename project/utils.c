#include "utils.h"
#include <stdio.h>


void getmul(const double* val, const double* vec, const int* rIndex, const int* cIndex, int nz, double* res)
{
	int i; 
	for (i = 0; i < nz; i++)
	{
		int rInd = rIndex[i];
		int cInd = cIndex[i];
		res[rInd] += val[i] * vec[cInd];
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


void dprintArray(int* a, int len){
	for (int i=0; i<len; i++)
        fprintf(stdout, "%d\t", a[i]);
    printf("\n");
}