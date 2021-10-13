#include <stdio.h> 

double integral(double a, double b, double (*f)(double), unsigned long int N)
{
  double sum=0, dx=(b-a)/N, x=a+dx;
  long int i;
  for(i=1; i<N; i++ )
 sum += f(x+i*dx);
  sum += (f(a)+f(b))/2;
  return sum*dx;
}

double polynomial(double x)  { return 2*x+1; }  

void main()  {
  long int N=5e8;
  double res, a=4, b=76; 
  res = integral(a,  b,  polynomial,  N);
  printf("I = %lf ",res);
}