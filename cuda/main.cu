#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <iostream>
#include <cuda_runtime.h>  // For CUDA runtime API
#include <helper_cuda.h>  // For checkCudaError macro
#include <helper_timer.h>  // For CUDA SDK timers

#include "lib/utils.h"

#define FILENAME "test-metrics.csv"



// What is a good initial guess for XBD and YBD (both
// greater than 1) ?
// After you get the code to work, experiment with different sizes
// to find the best possible performance
// Note: For meaningful time measurements you need sufficiently large matrices.
// Simple 1-D thread block
// Size should be at least 1 warp 
#define BLOCK_SIZE 256
#define WARP_SIZE        32

#define ERROR_BOUND 0.001


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














struct Result* module_cuda_csr(struct Csr* csr_mat, struct vector* vec)
{
    int *d_JA, *d_IRP;
    double *d_AS, *d_X, *d_res;
    double *res;

    int *IRP = csr_mat->IRP;
    int *JA = csr_mat->JA;
    double *AS = csr_mat->AS;
    double *X = vec->X;
    int M = csr_mat->M;
    int N = csr_mat->N;
    int nz = csr_mat->nz;


    printf("\n Start computation ... \n");
    

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
    checkCudaErrors(cudaMemcpy(d_JA, JA, nz*sizeof(int), cudaMemcpyHostToDevice));
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
    double elapsed_time = timer->getTime();
    double gpuflops=flopcnt/ elapsed_time;
    std::cout << "  GPU time: " << elapsed_time << " ms." << " GFLOPS " << gpuflops<<std::endl;

    // Download the resulting vector d_res from the device and store it in res.
    res = (double*)malloc(M*sizeof(double));
    memset(res, 0, M*sizeof(double));
    checkCudaErrors(cudaMemcpy(res, d_res, M*sizeof(double),cudaMemcpyDeviceToHost));

    delete timer;

    checkCudaErrors(cudaFree(d_AS));
    checkCudaErrors(cudaFree(d_JA));
    checkCudaErrors(cudaFree(d_IRP));
    checkCudaErrors(cudaFree(d_X));
    checkCudaErrors(cudaFree(d_res));

    struct Result* res_cuda_csr = (struct Result*) malloc(sizeof(struct Result));
    res_cuda_csr->res = res;
    res_cuda_csr->len = M;
    res_cuda_csr->elapsed_time = elapsed_time;
    res_cuda_csr->gpuflops = gpuflops;

    return res_cuda_csr;
}




struct Result* module_cuda_ellpack(struct Ellpack* ellpack_mat, struct vector* vec)
{
    int *d_JA_t, *d_MAXNZ;
    double *d_AS_t, *d_X, *d_res;
    double *res;

    double    *X = vec->X;
    double *AS_t = ellpack_mat->AS_t;
    int   *JA_t  = ellpack_mat->JA_t;
    int   *MAXNZ = ellpack_mat->MAXNZ;
    int        M = ellpack_mat->M;
    int        N = ellpack_mat->N;
    int        maxnz = ellpack_mat->maxnz;



    printf("\n Start computation ... \n");
    

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
    double elapsed_time = timer->getTime();
    double gpuflops=flopcnt/ elapsed_time;
    std::cout << "  GPU time: " << elapsed_time << " ms." << " GFLOPS " << gpuflops<<std::endl;

    // Download the resulting vector d_res from the device and store it in res.
    res = (double*)malloc(M*sizeof(double));
    memset(res, 0, M*sizeof(double));
    checkCudaErrors(cudaMemcpy(res, d_res, M*sizeof(double),cudaMemcpyDeviceToHost));



    struct Result* res_cuda_ellpack = (struct Result*) malloc(sizeof(struct Result));
    res_cuda_ellpack->res = res;
    res_cuda_ellpack->len = M;
    res_cuda_ellpack->elapsed_time = elapsed_time;
    res_cuda_ellpack->gpuflops = gpuflops;

    return res_cuda_ellpack;
}


int calculate_prod(struct matrix *mat, struct vector* vec, double *res_seq, char* mode, FILE *fpt)
{
    double *res;
    int     len;
    double elapsed_time;
    double gpuflops;
    int passed = 0;

    
    

    if( !strcmp(mode,"-cudaCSR")){

            struct Csr *csr_mat;

            printf("Start CUDA calculation with CSR...\n");

            csr_mat = preprocess_csr(mat);
            if( csr_mat == NULL )
                return -1;
            /* serial calculation with csr */
            struct Result *res_cuda_csr = module_cuda_csr(csr_mat, vec);

            res = res_cuda_csr->res;
            len = res_cuda_csr->len;
            elapsed_time = res_cuda_csr->elapsed_time;
            gpuflops = res_cuda_csr->gpuflops;
            

            free(csr_mat);



    }else if(!strcmp(mode,"-cudaELLPACK") ){
        
            struct Ellpack *ellpack_mat;

            printf("Start CUDA calculation with ELLPACK...\n");

            /* preprocess and build ellpack format */
            ellpack_mat = preprocess_ellpack(mat);
            if( ellpack_mat == NULL )
                return -1;
            /* calculation with ellpack with OpenMP */
            struct Result *res_cuda_ellpack = module_cuda_ellpack(ellpack_mat, vec);

            res = res_cuda_ellpack->res;
            len = res_cuda_ellpack->len;
            elapsed_time = res_cuda_ellpack->elapsed_time;
            gpuflops = res_cuda_ellpack->gpuflops;

            free(ellpack_mat);


    }

    double reldiff = 0.0f;
    double diff    = 0.0f;

    for (int i = 0; i < len; ++i) {
        double maxabs = std::max(std::abs(res_seq[i]),std::abs(res[i]));
        if (maxabs == 0.0) maxabs=1.0;
        reldiff = std::max(reldiff, std::abs(res_seq[i] - res[i])/maxabs);
        diff = std::max(diff, std::abs(res_seq[i] - res[i]));
    }
    std::cout << "Max diff = " << diff << "  Max rel diff = " << reldiff << std::endl;

    if( reldiff < ERROR_BOUND && diff < ERROR_BOUND )
        passed = 1;

    /* Print out the result in a file */
    fprintf(fpt,"%s, %lg, %lg, %d, %lg, %lg\n", mode, elapsed_time, gpuflops, passed, diff, reldiff);
    fflush(fpt);

    free(res);
    free(res_seq);    
    

    return 0;
}




int load_mat_vec(char* mat_filename, char* vector_filename, struct matrix* mat, struct vector* vec){

    int M, ret;

    /* preprocess matrix: from .mtx to matrix */
    ret = load_matrix(mat_filename, mat);
    if( ret == -1 )
        return -1;
    M = mat->M;

    /* load vector */
    ret = load_vector(vector_filename, vec, M);
    if( ret == -1 )
        return -1;

    return 0;
}



int main(int argc, char *argv[])
{
    
    int ret;
    FILE *fpt;
    double *res_seq;

    
    if (argc < 4)
    {
        fprintf(stderr, "Usage: %s [-cudaCSR/-cudaELLPACK] [martix-market-filename] [vector-filename]\n", argv[0]);
        exit(1);
    }

    //check if output file exists -> create
    if ( (fpt = fopen(FILENAME, "r")) == NULL) 
    {
        fpt = fopen(FILENAME, "w+");
        fprintf(fpt,"Matrix, M, N, nz, CalculationMode, CalculationTime(ms), GPUFlops, Passed\n");
        fflush(fpt);
    }
    fpt = fopen(FILENAME, "a");

    printf("Processing matrix %s\n\n", strrchr(argv[2], '/'));

    struct matrix* mat = (struct matrix*) malloc(sizeof(struct matrix));
    struct vector* vec = (struct vector*) malloc(sizeof(struct vector));

    if(!strcmp(argv[1],"-cudaCSR") ){

    }else if(!strcmp(argv[1],"-cudaELLPACK") ){

    }else{
        fprintf(stderr, "Usage: %s [-cudaCSR/-cudaELLPACK] [martix-market-filename] [vector-filename]\n", argv[0]);
        goto exit;
    }
    ret = load_mat_vec(argv[2], argv[3], mat, vec);
    if( ret == -1 ){
        fprintf(fpt,"\n");
        goto exit;
    }

    /* calculate the product result sequentially for testing */
    res_seq = (double*)malloc((mat->M)*sizeof(double));
    memset(res_seq, 0, (mat->M)*sizeof(double));

    getmul(mat, vec, res_seq);

    // write on file
    fprintf(fpt,"%s, %d, %d, %d, ", strrchr(argv[2], '/'), mat->M, mat->N, mat->nz);

    ret = calculate_prod(mat, vec, res_seq, argv[1], fpt);
    if( ret == -1 ){
        fprintf(fpt,"\n");
        goto exit;
    }

exit:
    fclose(fpt);
    free(mat);
    free(vec);

    


}