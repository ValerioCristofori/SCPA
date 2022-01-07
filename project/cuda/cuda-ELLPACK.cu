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




texture<float,1,cudaReadModeElementType> mainVecTexRef;

__device__ float ellpack_device(const float * AS, //values
                              const int * JA, //idx of column
                              const int maxEl,
                              const int row,
                              const int numRows)
{
    const int num_rows = numRows;
    //int maxEl = rowLength[row];
    float dot=0;   
    int col=-1;
    float val=0;
    int i=0;
    for(i=0; i<maxEl;i++)
    {
        col=JA[num_rows*i+row];
        val= AS[num_rows*i+row];
        dot+=val*tex1Dfetch(mainVecTexRef,col);
    }
    return dot;
}

__global__ void kernel_ellpack(const float * AS,
                                        const int * JA, 
                                        const int rowLength, 
                                        float * results,
                                        const int numRows)
{
    
    const int row   = blockDim.x * blockIdx.x + threadIdx.x;  // global thread index
    if(row<numRows)
    {
        float dot = ellpack_device(AS, JA, rowLength, row, numRows);
        results[row]=dot;
    }	
}





int main(int argc, char *argv[])
{
    int ret_code;
    MM_typecode matcode;
    FILE *f;
    int M, N, nz, xdim;  
    int i, *I, *J, *IRP;
    float *AS, *X, *res, *vIndex;
    int *d_JA, *d_IRP;
    float *d_AS, *d_res;

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

    if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) !=0)
        exit(1);


    /* reseve memory for matrices */

    I = (int *) malloc(nz * sizeof(int));
    J = (int *) malloc(nz * sizeof(int));
    AS = (float *) malloc(nz * sizeof(float));


    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (i=0; i<nz; i++)
    {
        fscanf(f, "%d %d %lg\n", &I[i], &J[i], &AS[i]);
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
    X = (float*)malloc(xdim * sizeof(float));
    for (i = 0; i<xdim; i++)
    {
        fscanf(f, "%lg\n", &X[i]);
    }

    if (f != stdin) fclose(f);


    /* preprocessing the matrix */
    //the original calculation result
    float* res_seq = (float*)malloc(M*sizeof(float));
    memset(res_seq, 0, M*sizeof(float));

    getmul(AS, X, I, J, nz, res_seq);

    //preprocess the dataset to make the calculation can be parallelized
    vIndex = (float*)malloc(nz*sizeof(float));
    memset(vIndex, 0, nz*sizeof(float));
    for (i = 0; i < nz; i++)
    {
        vIndex[i] = (float)I[i] * N + J[i];
        if (vIndex[i] < 0)
        {   
               printf("Error!\n");
               exit(1);
            }
    }

    quicksort(AS, vIndex, I, J, nz);





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

    // populate the 2D arrays
    int prev = 0;
    int count = 0;
    for ( int h = 0; h < nz; h++ )
    {
        x = I[h];

        if( prev == x ){
            count++;
        }else{
            // fill the rest of row with the latest value
            for( int k = 0; k < maxnz - count; k++ ){
                JA[prev*maxnz + count + k] = y;
            }
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




    float flopcnt=2.e-6*M*N;
    // Calculate the dimension of the grid of blocks (1D) needed to cover all
    // entries in the matrix and output vector
    const dim3 GRID_DIM(M,1);

    // setup data to send to the device
    checkCudaErrors(cudaMalloc((void**) &d_AS, (maxnz * M)*sizeof(float)));
    checkCudaErrors(cudaMalloc((void**) &d_JA, (maxnz * M)*sizeof(int)));
    checkCudaErrors(cudaMalloc((void**) &d_res, M*sizeof(float)));


    // copy arrays from the host (CPU) to the device (GPU)
    checkCudaErrors(cudaMemcpy(d_AS, AS, nz*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_JA, J, nz*sizeof(int), cudaMemcpyHostToDevice));

    // start timer
    // Create the CUDA SDK timer.
    StopWatchInterface* timer = 0;
    sdkCreateTimer(&timer);

    timer->start();

    kernel_ellpack<<<GRID_DIM, BLOCK_DIM>>>(d_AS, d_JA, maxnz, d_res, M);
    checkCudaErrors(cudaDeviceSynchronize());

    timer->stop();
    float gpuflops=flopcnt/ timer->getTime();
    std::cout << "  GPU time: " << timer->getTime() << " ms." << " GFLOPS " << gpuflops<<std::endl;

    // Download the resulting vector d_res from the device and store it in res.
    res = (float*)malloc(M*sizeof(float));
    memset(res, 0, M*sizeof(float));
    checkCudaErrors(cudaMemcpy(res, d_res, M*sizeof(float),cudaMemcpyDeviceToHost));


    // check the result if the same
    if (!checkerror(res, res_seq, M))
    {
        printf("Calculation Error!\n");
        exit(1);
    }
    else {
        printf(" Test Result Passed ... \n");
    }

    /* Print out the result in a file -> output.txt */

    if ((f = fopen("output.txt", "w")) == NULL)
    {
        printf("Fail to open the output file!\n");
        exit(1);
    }
    for (i = 0; i<xdim; i++)
    {
        fprintf(f, "%lg\n", res[i]);
    }
    fclose(f);


    // free all

    delete timer;

    checkCudaErrors(cudaFree(d_AS));
    checkCudaErrors(cudaFree(d_JA));
    checkCudaErrors(cudaFree(d_res));

    free(res_seq);
    free(X);
    free(vIndex);
    free(JA);
    free(AS);
    free(I);
    free(J);
    free(res);


}