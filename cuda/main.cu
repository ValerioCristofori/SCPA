#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <iostream>
#include <cuda_runtime.h>  // For CUDA runtime API
#include <helper_cuda.h>   // For checkCudaError macro
#include <helper_timer.h>  // For CUDA SDK timers

#include "lib/utils.h"

#define FILENAME "test-metrics.csv" // output filename

// Change to find the best possible performance
// Note: For meaningful time measurements you need sufficiently large matrices.
#ifndef BLOCK_SIZE
#define BLOCK_SIZE      256
#endif

#define WARP_SIZE       32



const dim3 BLOCK_DIM(BLOCK_SIZE);


__global__ void kernel_csr(const double * AS, //values
                                    const int * JA, //idx of column
                                    const int * IRP, //ptrs to rows
                                    const double * X, //vector for the product
                                    double * results,
                                    const int num_rows)
{
    __shared__ double   sdata[ BLOCK_SIZE + 16 ];          // padded to avoid reduction ifs
    __shared__ int       ptrs[ BLOCK_SIZE / WARP_SIZE ][2];
    
    const int thread_id   = BLOCK_SIZE * blockIdx.x + threadIdx.x;    // global thread index
    const int thread_lane = threadIdx.x & ( WARP_SIZE - 1);           // thread index within the warp
    const int warp_id     = thread_id   /  WARP_SIZE ;                // global warp index
    const int warp_lane   = threadIdx.x /  WARP_SIZE ;                // warp index within the CTA
    const int num_warps   = ( BLOCK_SIZE  /  WARP_SIZE ) * gridDim.x; // total number of active warps

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
            results[row] = smem[threadIdx.x];
        }
    }
}



/* JA and AS are trasposed */
__device__ double ellpack_device(const double * AS,
                              const int * JA,
                              const double * X,
                              const int * MAXNZ,
                              const int row,
                              const int numRows)
{
    const int num_rows = numRows;
    int          maxnz = MAXNZ[row];
    double         dot = 0;   
    int            col = -1;
    double         val = 0;
    int              i = 0;
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
    // one thread calculate prod for one row
    if(row<numRows)
    {
        double dot = ellpack_device(AS,JA,X,MAXNZ,row,numRows);
        results[row]=dot;
    }   
}














struct Result* module_cuda_csr(struct Csr* csr_mat, struct vector* vec)
{
    int     *d_JA, *d_IRP;
    double  *d_AS, *d_X, *d_res;
    double  *res;

    int     *IRP = csr_mat->IRP;
    int      *JA = csr_mat->JA;
    double   *AS = csr_mat->AS;
    double    *X = vec->X;
    int        M = csr_mat->M;
    int        N = csr_mat->N;
    int       nz = csr_mat->nz;


    printf("\n Start computation ... \n");
    

    double flopcnt=2.e-6*nz;
    // Calculate the dimension of the grid of blocks (1D) needed to cover all
    // entries in the matrix and output vector
    const dim3 GRID_DIM((M - 1 + BLOCK_DIM.x)/ BLOCK_DIM.x  ,1);

    // setup data to send to the device
    checkCudaErrors(cudaMalloc((void**) &d_AS, nz*sizeof(double)));
    checkCudaErrors(cudaMalloc((void**) &d_JA, nz*sizeof(int)));
    checkCudaErrors(cudaMalloc((void**) &d_IRP, (M+1)*sizeof(int)));
    checkCudaErrors(cudaMalloc((void**) &d_X, N*sizeof(double)));
    checkCudaErrors(cudaMalloc((void**) &d_res, M*sizeof(double)));


    // copy arrays from the host (CPU) to the device (GPU)
    checkCudaErrors(cudaMemcpy(d_AS, AS, nz*sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_JA, JA, nz*sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_IRP, IRP, (M+1)*sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_X, X, N*sizeof(double), cudaMemcpyHostToDevice));


    // start timer
    // Create the CUDA SDK timer.
    StopWatchInterface* timer = 0;
    sdkCreateTimer(&timer);

    timer->start();

    //-------------------------- call Kernel -----------------------------------------
    kernel_csr<<<GRID_DIM, BLOCK_DIM>>>(d_AS, d_JA, d_IRP, d_X, d_res, M);
    checkCudaErrors(cudaDeviceSynchronize());

    timer->stop();
    double elapsed_time = timer->getTime();
    double gpuflops=flopcnt/ elapsed_time;
    std::cout << "  GPU time: " << elapsed_time << " ms." << " GFLOPS " << gpuflops<<std::endl;

    /* allocate memory for result vector 
        download the resulting vector d_res 
        from the device and store it in res. */
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
    int     *d_JA_t, *d_MAXNZ;
    double  *d_AS_t, *d_X, *d_res;
    double  *res;

    double    *X = vec->X;
    double *AS_t = ellpack_mat->AS_t;
    int   *JA_t  = ellpack_mat->JA_t;
    int   *MAXNZ = ellpack_mat->MAXNZ;
    int        M = ellpack_mat->M;
    int        N = ellpack_mat->N;
    int       nz = ellpack_mat->nz;
    int    maxnz = ellpack_mat->maxnz;



    printf("\n Start computation ... \n");
    

    double flopcnt=2.e-6*nz;
    // Calculate the dimension of the grid of blocks (1D) needed to cover all
    // entries in the matrix and output vector
    const dim3 GRID_DIM((M - 1 + BLOCK_DIM.x)/ BLOCK_DIM.x  ,1);

    // setup data to send to the device
    checkCudaErrors(cudaMalloc((void**) &d_AS_t, (maxnz*M)*sizeof(double)));
    checkCudaErrors(cudaMalloc((void**) &d_JA_t, (maxnz*M)*sizeof(int)));
    checkCudaErrors(cudaMalloc((void**) &d_MAXNZ, M*sizeof(int)));
    checkCudaErrors(cudaMalloc((void**) &d_X, N*sizeof(double)));
    checkCudaErrors(cudaMalloc((void**) &d_res, M*sizeof(double)));


    // copy arrays from the host (CPU) to the device (GPU)
    checkCudaErrors(cudaMemcpy(d_AS_t, AS_t, (maxnz*M)*sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_JA_t, JA_t, (maxnz*M)*sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_MAXNZ, MAXNZ, M*sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_X, X, N*sizeof(double), cudaMemcpyHostToDevice));


    // start timer
    // Create the CUDA SDK timer.
    StopWatchInterface* timer = 0;
    sdkCreateTimer(&timer);

    timer->start();

    //-------------------------- call Kernel -----------------------------------------
    kernel_ellpack<<<GRID_DIM, BLOCK_DIM>>>(d_AS_t, d_JA_t, d_MAXNZ, d_X, d_res, M);
    checkCudaErrors(cudaDeviceSynchronize());

    timer->stop();
    double elapsed_time = timer->getTime();
    double gpuflops=flopcnt/ elapsed_time;
    std::cout << "  GPU time: " << elapsed_time << " ms." << " GFLOPS " << gpuflops<<std::endl;

    /* allocate memory for result vector 
        download the resulting vector d_res 
        from the device and store it in res. */    
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


int calculate_prod(struct matrix *mat, struct vector* vec, double *res_seq, int dim_res_seq, char* mode, FILE *fpt)
{
    double      *res;           // result of the parallel product
    int         len;            // len of the result
    double      elapsed_time;   // time spent in the calculation
    double      gpuflops;       // GPU floating point ops per second
    int         passed = 0;     // 1 if the parallelized product is successful 
    struct Result *result;

    double reldiff      = 0.0f; //
    double diff         = 0.0f; //

    /* select the right matrix preprocessing and calculation mode
      with respect to the 'mode' value entered   */
    if( !strcmp(mode,"-cudaCSR")){

            struct Csr      *csr_mat;

            printf("Start CUDA calculation with CSR...\n");

            /* pre-processing the matrix following CSR format */
            csr_mat = preprocess_csr(mat);
            if( csr_mat == NULL )
                return -1;
            
            /* serial calculation with csr */
            result = module_cuda_csr(csr_mat, vec);

            free(csr_mat);



    }else if(!strcmp(mode,"-cudaELLPACK") ){
        
            struct Ellpack      *ellpack_mat;

            printf("Start CUDA calculation with ELLPACK...\n");

            /* preprocess and build ellpack format for the matrix */
            ellpack_mat = preprocess_ellpack(mat);
            if( ellpack_mat == NULL )
                return -1;
            
            /* calculation with ellpack with OpenMP */
            result = module_cuda_ellpack(ellpack_mat, vec);

            free(ellpack_mat);

    }

    // check if parallel calculation is successful done
    if (!checkerror(result, res_seq, dim_res_seq))
    {
        printf("Calculation Error!\n");

    }
    else {
        printf(" Test Result Passed ... \n");
        passed = 1;
    }

    // build vars for verification and metrics
    elapsed_time = result->elapsed_time;
    gpuflops = result->gpuflops;
    diff     = result->diff;
    reldiff  = result->reldiff;

    
    

    /* Print out the result in a file */
    fprintf(fpt,"%s,%d,%lg,%lg,%d,%lg,%lg\n", mode, BLOCK_SIZE, elapsed_time, gpuflops, passed, diff, reldiff);
    fflush(fpt);

    free(result->res);
    free(result);
    free(res_seq);    
    

    return 0;
}




int load_mat_vec(char* mat_filename, char* vector_filename, struct matrix* mat, struct vector* vec){

    int N, ret;

    /* preprocess matrix: from .mtx to matrix */
    ret = load_matrix(mat_filename, mat);
    if( ret == -1 )
        return -1;
    N = mat->N;

    /* load vector from .txt: generate with 'vector_generator.sh' */
    /* first entry of the file provide vector's length */
    ret = load_vector(vector_filename, vec, N);
    if( ret == -1 )
        return -1;

    return 0;
}



int main(int argc, char *argv[])
{
    
    int     ret;   // return code
    FILE    *fpt;  // file ptr use to collect metrics of calculation
    double  *res_seq;

    
    if (argc < 4)
    {
        fprintf(stderr, "Usage: %s [-cudaCSR/-cudaELLPACK] [martix-market-filename] [vector-filename]\n", argv[0]);
        exit(1);
    }

    //check if output file exists -> create
    if ( (fpt = fopen(FILENAME, "r")) == NULL) 
    {
        // create the output file and a '.csv' header
        fpt = fopen(FILENAME, "w+");
        fprintf(fpt,"Matrix,M,N,nz,CalculationMode,Threads,CalculationTime(ms),GFlops,Passed,Diff,RelDiff\n");
        fflush(fpt);
    }
    // open file in append mode for add new entry
    fpt = fopen(FILENAME, "a");

    printf("Processing matrix %s\n\n", strrchr(argv[2], '/'));

    /* allocate memory for matrix and vector struct */
    struct matrix* mat = (struct matrix*) malloc(sizeof(struct matrix));
    struct vector* vec = (struct vector*) malloc(sizeof(struct vector));

    /* check if 'argv[1]' match one calc mode */ 
    if(!strcmp(argv[1],"-cudaCSR") ){


    }else if(!strcmp(argv[1],"-cudaELLPACK") ){


    }else{
        
        fprintf(stderr, "Usage: %s [-cudaCSR/-cudaELLPACK] [martix-market-filename] [vector-filename]\n", argv[0]);
        goto exit;
    }

    /* load matrix and vector to product building the structs */
    ret = load_mat_vec(argv[2], argv[3], mat, vec);
    if( ret == -1 ){
        // some error occurs -> prepare output file for a new run
        fprintf(fpt,"\n");
        goto exit;
    }

    /* allocate memory for the result of the sequential product*/
    res_seq = (double*)malloc((mat->M)*sizeof(double));
    memset(res_seq, 0, (mat->M)*sizeof(double));

    /* calculate the product result sequentially for testing */
    /* the result in 'res_seq' must be compared with 
      the result that comes out of the parallelization */
    getmul(mat, vec, res_seq);

    // write on 'csv' output file the matrix informations
    fprintf(fpt,"%s,%d,%d,%d,", strrchr(argv[2], '/'), mat->M, mat->N, mat->nz);

    /* call the function responsible for calculating 
    the product according to the mode entered */
    ret = calculate_prod(mat, vec, res_seq, mat->M, argv[1], fpt);
    if( ret == -1 ){
        fprintf(fpt,"\n");
        goto exit;
    }

exit:
    fclose(fpt);
    free(mat);
    free(vec);

}