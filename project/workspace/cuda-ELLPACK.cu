#include <iostream>

#include <cuda_runtime.h>  // For CUDA runtime API
#include <helper_cuda.h>  // For checkCudaError macro
#include <helper_timer.h>  // For CUDA SDK timers
#include "mmio.h"
#include "utils.h"

// What is a good initial guess for XBD and YBD (both
// greater than 1) ?
// After you get the code to work, experiment with different sizes
// to find the best possible performance
// Note: For meaningful time measurements you need sufficiently large matrices.
// Simple 1-D thread block
// Size should be at least 1 warp 
#define BLOCK_SIZE 256
#define WARP_SIZE        32

const dim3 BLOCK_DIM(BLOCK_SIZE);


__device__ double ellpack_device(const double * AS,
                              const int * JA,
                              const double * X,
                              const int * MAXNZ,
                              const int row,
                              const int numRows)
{
    const int num_rows =numRows;
    int maxnz = MAXNZ[row];
    double dot=0;   
    int col=-1;
    double val=0;
    int i=0;
    for(i=0; i<maxnz;i++)
    {
        col=JA[num_rows*i+row];
        val= AS[num_rows*i+row];
        dot+=val*X[col];
        /*col=JA[row*maxnz+i]; //non puo' funzionare perche' maxnz cambia e moltiplica male
        val= AS[row*maxnz+i];
        dot+=val*X[col];*/
    }
    return dot;
}

__global__ void kernel_ellpack(const double * AS,
                                        const int * JA, 
                                        const int * MAXNZ,
                                        const double * X,
                                        double * results,
                                        const int numRows)
{
    
    const int row   = blockDim.x * blockIdx.x + threadIdx.x;  // global thread index
    if(row<numRows)
    {
        double dot = ellpack_device(AS,JA,X,MAXNZ,row,numRows);
        results[row]=dot;
    }   
}



/*__device__ double ellpack_device(const double * AS,
                              const int * JA,
                              const double * X,
                              const int maxEl,
                              const int row)
{
    //int maxEl = rowLength[row];
    double dot=0;   
    int col=-1;
    double val=0;
    int i=0;
    for(i=0; i<maxEl;i++)
    {
        col=JA[maxEl*row+i];
        val= AS[maxEl*row+i];
        dot+=val*X[col];
    }
    return dot;
}

__global__ void kernel_ellpack(const double * AS, //values
                                        const int * JA, //idx of column 
                                        const int rowLength,
                                        const double * X, //vector for the product 
                                        double * results,
                                        const int numRows)
{
    
    const int row   = blockDim.x * blockIdx.x + threadIdx.x;  // global thread index
    if(row<numRows)
    {
        double dot = ellpack_device(AS, JA, X, rowLength, row);
        results[row]=dot;
    }	
}
*/




int main(int argc, char *argv[])
{
    MM_typecode matcode;
    FILE *f;
    int M, N, nz, xdim;
    int  x = 0, y = 0, offset = 0, maxnz = 0;
    int i, j, *I, *J, *JA, *JA_t, *MAXNZ;
    double *AS, *AS_t, *X, *res, *vIndex, *val;
    double value = 0.0;
    
    int *d_JA_t, *d_MAXNZ;
    double *d_AS_t, *d_X, *d_res;

    if (argc < 3)
    {
        fprintf(stderr, "Usage: %s [martix-market-filename] [vector-filename]\n", argv[0]);
        exit(1);
    }
    else    
    { 
        if ((f = fopen(argv[1], "r")) == NULL) 
            exit(1);
    }

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        exit(1);
    }


    /*  This is how one can screen matrix types if their application */
    /*  only supports a subset of the Matrix Market data types.      */

    if (mm_is_complex(matcode) && mm_is_matrix(matcode) && 
            mm_is_sparse(matcode) )
    {
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        exit(1);
    }

    /* find out size of sparse matrix */

    if ( mm_read_mtx_crd_size(f, &M, &N, &nz) !=0)
        exit(1);


    /* reseve memory for matrices */

    I = (int *) malloc(nz * sizeof(int));
    J = (int *) malloc(nz * sizeof(int));
    val = (double *) malloc(nz * sizeof(double));


    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (i=0; i<nz; i++)
    {
        fscanf(f, "%d %d %lg\n", &I[i], &J[i], &val[i]);
        I[i]--;  /* adjust from 1-based to 0-based */
        J[i]--;
    }

    if (f !=stdin) fclose(f);


    /* Open and load the vector input for the product */  
    if ((f = fopen(argv[2], "r")) == NULL)
    {
        printf("Fail to open the input vector file!\n");
        exit(1);
    }
    fscanf(f, "%d\n", &xdim);
    if (xdim > M)
    {
        xdim = M;
    } else {
        printf("dimension vector too small!\n");
        exit(1);
    }
    X = (double*)malloc(xdim * sizeof(double));
    for (i = 0; i<xdim; i++)
    {
        fscanf(f, "%lg\n", &X[i]);
    }

    if (f != stdin) fclose(f);


    /* preprocessing the matrix */
    //the original calculation result
    double* res_seq = (double*)malloc(M*sizeof(double));
    memset(res_seq, 0, M*sizeof(double));

    getmul(val, X, I, J, nz, res_seq);

    //preprocess the dataset to make the calculation can be parallelized
    vIndex = (double*)malloc(nz*sizeof(double));
    memset(vIndex, 0, nz*sizeof(double));
    for (i = 0; i < nz; i++)
    {
        vIndex[i] = (double)I[i] * N + J[i];
        if (vIndex[i] < 0)
        {   
               printf("Error!\n");
               exit(1);
            }
    }

    quicksort(val, vIndex, I, J, nz);


    int count_nz = 1;
    for (i = 0; i<nz-1; i++)
    {
        if ( I[i] == I[i+1] ){
            count_nz++;
        }else{
            if( count_nz > maxnz )
                maxnz = count_nz;
            count_nz = 1;
        }

    }
    if( count_nz > maxnz )
        maxnz = count_nz;
    if ( maxnz > nz )
    {
        printf("MaxNZ Error!\n");
        exit(1);
    }

    
    // reserve JA and AS 2D arrays
    JA = (int *) malloc((maxnz * M) * sizeof(int));
    memset(JA, -1, (maxnz*M)*sizeof(int));
    AS = (double *) malloc((maxnz * M) * sizeof(double));
    memset(AS, 0, (maxnz*M)*sizeof(double));
    MAXNZ = (int *) malloc( M* sizeof(int));
    memset(MAXNZ, -1, M*sizeof(int));


    // populate the 2D arrays
    int prev = 0;
    int count = 0;
    int count_row = 0;
    offset = 0;
    for ( int h = 0; h < nz; h++ )
    {
        x = I[h];

        if( prev == x ){
            count++;
        }else{
            //new row
            // fill the rest of row with the latest value
            for( int k = 0; k < maxnz - count; k++ ){
                JA[prev*maxnz + count + k] = y;
            }
            MAXNZ[count_row] = count;
            count_row++;
            offset += count; //add prev row length to the offset 

            count = 0;

            prev = x;
            h--;
            continue;
        }

        prev = x;
        y = J[h];
        value = val[h];
        JA[x*maxnz + count - 1] = y;
        AS[x*maxnz + count - 1] = value;

    }
    // lastline
    while( count < maxnz ){
        JA[x*maxnz + count] = y;
        count++;
    }
    MAXNZ[count_row] = count;

    //transposition vector AS e JA
    //reserve memory
    JA_t = (int *) malloc((maxnz*M) * sizeof(int));
    memset(JA_t, -1, (maxnz*M)*sizeof(int));
    AS_t = (double *) malloc((maxnz*M) * sizeof(double));
    memset(AS_t, 0, (maxnz*M)*sizeof(double));

    offset = 0;
    for(i=0; i<M; i++)
    {
        for(j=0; j<maxnz; j++)
        {
            JA_t[j*M+i]=JA[i*maxnz+j];
            AS_t[j*M+i]=AS[i*maxnz+j];
        }
    }

    /*printf("AS:\n");
    dprintArrayDouble(AS, maxnz*M);
    printf("\n\n");
    printf("JA:\n");
    dprintArrayInt(JA, maxnz*M);
    printf("\n\n");
    printf("AS trasposto:\n");
    dprintArrayDouble(AS_t, maxnz*M);
    printf("\n\n");
    printf("JA trasposto:\n");
    dprintArrayInt(JA_t, maxnz*M);
    printf("\n\n");
    printf("X:\n");
    dprintArrayDouble(X, M);
    printf("\n\n");
    dprintArrayInt(MAXNZ, M);*/


    double flopcnt=2.e-6*M*N;
    // Calculate the dimension of the grid of blocks (1D) needed to cover all
    // entries in the matrix and output vector
    const dim3 GRID_DIM((M - 1 + BLOCK_DIM.x)/ BLOCK_DIM.x  ,1);

    // setup data to send to the device
    checkCudaErrors(cudaMalloc((void**) &d_AS_t, (maxnz*M)*sizeof(double)));
    checkCudaErrors(cudaMalloc((void**) &d_JA_t, (maxnz*M)*sizeof(int)));
    checkCudaErrors(cudaMalloc((void**) &d_MAXNZ, M*sizeof(int)));
    checkCudaErrors(cudaMalloc((void**) &d_X, M*sizeof(double)));
    checkCudaErrors(cudaMalloc((void**) &d_res, M*sizeof(double)));


    // copy arrays from the host (CPU) to the device (GPU)
    checkCudaErrors(cudaMemcpy(d_AS_t, AS_t, (maxnz*M)*sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_JA_t, JA_t, (maxnz*M)*sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_MAXNZ, MAXNZ, M*sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_X, X, M*sizeof(double), cudaMemcpyHostToDevice));


    // start timer
    // Create the CUDA SDK timer.
    StopWatchInterface* timer = 0;
    sdkCreateTimer(&timer);

    timer->start();

    kernel_ellpack<<<GRID_DIM, BLOCK_DIM>>>(d_AS_t, d_JA_t, d_MAXNZ, d_X, d_res, M);
    checkCudaErrors(cudaDeviceSynchronize());

    timer->stop();
    double gpuflops=flopcnt/ timer->getTime();
    std::cout << "  GPU time: " << timer->getTime() << " ms." << " GFLOPS " << gpuflops<<std::endl;

    // Download the resulting vector d_res from the device and store it in res.
    res = (double*)malloc(M*sizeof(double));
    memset(res, 0, M*sizeof(double));
    checkCudaErrors(cudaMemcpy(res, d_res, M*sizeof(double),cudaMemcpyDeviceToHost));

    /*dprintArrayDouble(res, M);
    printf("\n\n");
    dprintArrayDouble(res_seq, M);*/

    // Now let's check if the results are the same.
    double reldiff = 0.0f;
    double diff = 0.0f;
  
    for (i = 0; i < M; ++i) {
        double maxabs = std::max(std::abs(res_seq[i]),std::abs(res[i]));
        if (maxabs == 0.0) maxabs=1.0;
        reldiff = std::max(reldiff, std::abs(res_seq[i] - res[i])/maxabs);
        diff = std::max(diff, std::abs(res_seq[i] - res[i]));
    }
    std::cout << "Max diff = " << diff << "  Max rel diff = " << reldiff << std::endl;

    /* Print out the result in a file -> output.txt */

    if ((f = fopen("output.txt", "w")) == NULL)
    {
        printf("Fail to open the output file!\n");
        goto exit;
    }
    for (i = 0; i<xdim; i++)
    {
        fprintf(f, "%lg\n", res[i]);
    }
    fclose(f);

exit:
    // free all

    delete timer;

    checkCudaErrors(cudaFree(d_AS_t));
    checkCudaErrors(cudaFree(d_JA_t));
    checkCudaErrors(cudaFree(d_X));
    checkCudaErrors(cudaFree(d_res));

    free(res_seq);
    free(X);
    free(vIndex);
    free(val);
    free(JA);
    free(AS);
    free(JA_t);
    free(AS_t);
    free(I);
    free(J);
    free(res);


}