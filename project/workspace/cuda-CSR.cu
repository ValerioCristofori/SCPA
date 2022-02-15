#include <iostream>

#include <cuda_runtime.h>  // For CUDA runtime API
#include <helper_cuda.h>  // For checkCudaError macro
#include <helper_timer.h>  // For CUDA SDK timers
#include "libraries/mmio.h"
#include "libraries/utils.h"

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


__global__ void kernel_csr(const double * AS, //values
                                    const int * JA, //idx of column
                                    const int * IRP, //ptrs to rows
                                    const double * X, //vector for the product
                                    double * results,
                                    const int num_rows)
{
    __shared__ double sdata[ BLOCK_SIZE + 16];                    // padded to avoid reduction ifs
    __shared__ int ptrs[ BLOCK_SIZE / WARP_SIZE ][2];
    
    const int thread_id   = BLOCK_SIZE * blockIdx.x + threadIdx.x;  // global thread index
    const int thread_lane = threadIdx.x & ( WARP_SIZE - 1);            // thread index within the warp
    const int warp_id     = thread_id   /  WARP_SIZE ;                // global warp index
    const int warp_lane   = threadIdx.x /  WARP_SIZE ;                // warp index within the CTA
    const int num_warps   = ( BLOCK_SIZE  /  WARP_SIZE ) * gridDim.x;   // total number of active warps

    for(int row = warp_id; row < num_rows; row += num_warps){
        // use two threads to fetch IRP[row] and IRP[row+1] // vector pointers
        // this is considerably faster than the straightforward version
        if(thread_lane < 2)
            ptrs[warp_lane][thread_lane] = IRP[row + thread_lane];
        const int row_start = ptrs[warp_lane][0];            //same as: row_start = IRP[row];
        const int row_end   = ptrs[warp_lane][1];            //same as: row_end   = IRP[row+1];

        // compute local sum
        double sum = 0;
        for(int j = row_start + thread_lane; j < row_end; j += WARP_SIZE)
        {
            sum += AS[j] * X[JA[j]];
        }

        volatile double* smem = sdata;
        smem[threadIdx.x] = sum; __syncthreads(); 
        smem[threadIdx.x] = sum = sum + smem[threadIdx.x + 16];
        smem[threadIdx.x] = sum = sum + smem[threadIdx.x +  8];
        smem[threadIdx.x] = sum = sum + smem[threadIdx.x +  4];
        smem[threadIdx.x] = sum = sum + smem[threadIdx.x +  2];
        smem[threadIdx.x] = sum = sum + smem[threadIdx.x +  1];

        // first thread writes warp result
        if (thread_lane == 0){
            results[row]=smem[threadIdx.x];
        }
    }
}



int main(int argc, char *argv[])
{
    MM_typecode matcode;
    FILE *f;
    int M, N, nz, xdim;  
    int i, *I, *J, *IRP;
    double *AS, *X, *res, *vIndex;
    int *d_JA, *d_IRP;
    double *d_AS, *d_X, *d_res;

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
    AS = (double *) malloc(nz * sizeof(double));


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

    getmul(AS, X, I, J, nz, res_seq);

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

    quicksort(AS, vIndex, I, J, nz);

    IRP = (int*)malloc((M+1)*sizeof(int)); //start position of each row
    memset(IRP, -1, (M+1)*sizeof(int));

    for (i = 0; i<nz; i++)
    {
        int tmp = (int)(vIndex[i] / N);
        if (IRP[tmp] == -1)
        {
            IRP[tmp] = i;
        }

    }
    // update last entry in IRP array with the greater one
    IRP[M] = nz + 1;

    double flopcnt=2.e-6*M*N;
    // Calculate the dimension of the grid of blocks (1D) needed to cover all
    // entries in the matrix and output vector
    const dim3 GRID_DIM((M - 1 + BLOCK_DIM.x)/ BLOCK_DIM.x  ,1);

    // setup data to send to the device
    checkCudaErrors(cudaMalloc((void**) &d_AS, nz*sizeof(double)));
    checkCudaErrors(cudaMalloc((void**) &d_JA, nz*sizeof(int)));
    checkCudaErrors(cudaMalloc((void**) &d_IRP, (M+1)*sizeof(int)));
    checkCudaErrors(cudaMalloc((void**) &d_X, M*sizeof(double)));
    checkCudaErrors(cudaMalloc((void**) &d_res, M*sizeof(double)));


    // copy arrays from the host (CPU) to the device (GPU)
    checkCudaErrors(cudaMemcpy(d_AS, AS, nz*sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_JA, J, nz*sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_IRP, IRP, (M+1)*sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_X, X, M*sizeof(double), cudaMemcpyHostToDevice));


    // start timer
    // Create the CUDA SDK timer.
    StopWatchInterface* timer = 0;
    sdkCreateTimer(&timer);

    timer->start();

    kernel_csr<<<GRID_DIM, BLOCK_DIM>>>(d_AS, d_JA, d_IRP, d_X, d_res, M);
    checkCudaErrors(cudaDeviceSynchronize());

    timer->stop();
    double gpuflops=flopcnt/ timer->getTime();
    std::cout << "  GPU time: " << timer->getTime() << " ms." << " GFLOPS " << gpuflops<<std::endl;

    // Download the resulting vector d_res from the device and store it in res.
    res = (double*)malloc(M*sizeof(double));
    memset(res, 0, M*sizeof(double));
    checkCudaErrors(cudaMemcpy(res, d_res, M*sizeof(double),cudaMemcpyDeviceToHost));


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

    checkCudaErrors(cudaFree(d_AS));
    checkCudaErrors(cudaFree(d_JA));
    checkCudaErrors(cudaFree(d_IRP));
    checkCudaErrors(cudaFree(d_X));
    checkCudaErrors(cudaFree(d_res));

    free(res_seq);
    free(vIndex);
    free(res);
    free(AS);
    free(I);
    free(J);
    free(X);
    free(IRP);
}