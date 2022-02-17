#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include <iostream>
#include <cuda_runtime.h>  // For CUDA runtime API
#include <helper_cuda.h>  // For checkCudaError macro
#include <helper_timer.h>  // For CUDA SDK timers

#include "lib/mmio.h"
#include "lib/utils.h"



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








struct Csr* preprocess_csr(struct matrix *mat)
{
    struct Csr *csr_mat;

    int M = mat->M;
    int N = mat->N;
    int nz = mat->nz;
    int *I = mat->I;
    int *J = mat->J;
    double *val = mat->val;

    //preprocess the dataset to make the calculation can be parallelized
    double *vIndex = (double*)malloc(nz*sizeof(double));
    memset(vIndex, 0, nz*sizeof(double));
    for (int i = 0; i < nz; i++)
    {
        vIndex[i] = (double)I[i] * N + J[i];
        if (vIndex[i] < 0)
        {   
               printf("Error: %lg < 0\n", vIndex[i]);
               return NULL;
            }
    }

    quicksort(val, vIndex, I, J, nz);

    int *IRP = (int*)malloc((M+1)*sizeof(int)); //start position of each row
    memset(IRP, -1, (M+1)*sizeof(int));


    for (int i = 0; i<nz; i++)
    {
        int tmp = (int)(vIndex[i] / N);
        if (IRP[tmp] == -1)
        {
            IRP[tmp] = i;
        }

    }
    // update last entry in IRP array with the greater one
    IRP[M] = nz;

    csr_mat = (struct Csr*) malloc(sizeof(struct Csr));
    csr_mat->M = M;
    csr_mat->N = N;
    csr_mat->JA = J;
    csr_mat->AS = val;
    csr_mat->IRP = IRP;

    free(vIndex);

    return csr_mat;
}



struct Ellpack* preprocess_ellpack(struct matrix *mat)
{

    struct Ellpack *ellpack_mat;

    int *JA_t;
    double *AS_t;

    int maxnz = 0;
    int M = mat->M;
    int N = mat->N;
    int nz = mat->nz;
    int *I = mat->I;
    int *J = mat->J;
    double *val = mat->val;

    //preprocess the dataset to make the calculation can be parallelized
    double *vIndex = (double*)malloc(nz*sizeof(double));
    memset(vIndex, 0, nz*sizeof(double));
    for (int i = 0; i < nz; i++)
    {
        vIndex[i] = (double)I[i] * N + J[i];
        if (vIndex[i] < 0)
        {   
               printf("Error: %lg < 0\n", vIndex[i]);
               return NULL;
            }
    }

    quicksort(val, vIndex, I, J, nz);

    free(vIndex);
    
    int count_nz = 1;
    for (int i = 0; i<nz-1; i++)
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
    int *JA = (int *) malloc((maxnz * M) * sizeof(int));
    memset(JA, 0, (maxnz*M)*sizeof(int));
    double *AS = (double *) malloc((maxnz * M) * sizeof(double));
    memset(AS, 0, (maxnz*M)*sizeof(double));
    int *MAXNZ = (int *) malloc( M* sizeof(int));
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

    ellpack_mat = (struct Ellpack*) malloc(sizeof(struct Ellpack));
    ellpack_mat->M = M;
    ellpack_mat->N = N;
    ellpack_mat->MAXNZ = MAXNZ;
    ellpack_mat->JA_t = JA_t;
    ellpack_mat->AS_t = AS_t;

    return ellpack_mat;
}






struct Result* module_cuda_csr(struct Csr* csr_mat, struct vector* vec)
{
    int j, ckey;
    int *d_JA, *d_IRP;
    double *d_AS, *d_X, *d_res;


    int *IRP = csr_mat->IRP;
    int *JA = csr_mat->JA;
    double *AS = csr_mat->AS;
    double *X = vec->X;
    int M = csr_mat->M;
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
    long elapsed_time = timer->getTime();
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
    int ja;

    int *d_JA_t, *d_MAXNZ;
    double *d_AS_t, *d_X, *d_res;

    double    *X = vec->X;
    double *AS_t = ellpack_mat->AS_t;
    int   *JA_t  = ellpack_mat->JA_t;
    int   *MAXNZ = ellpack_mat->MAXNZ;
    int        M = ellpack_mat->M;



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
    long elapsed_time = timer->getTime();
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
    long elapsed_time;
    double gpuflops;
    int passed = 0;

    
    

    if( !strcmp(mode,"-cudaCSR")){

            struct Csr *csr_mat;

            printf("Start CUDA calculation with CSR...\n");

            csr_mat = preprocess_csr(mat);
            if( csr_mat == NULL )
                return 1;
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
                return 1;
            /* calculation with ellpack with OpenMP */
            struct Result *res_cuda_ellpack = module_cuda_ellpack(ellpack_mat, vec);

            res = res_cuda_ellpack->res;
            len = res_cuda_ellpack->len;
            elapsed_time = res_cuda_ellpack->elapsed_time;
            gpuflops = res_cuda_ellpack->gpuflops;

            free(ellpack_mat);


    }

    for (int i = 0; i < len; ++i) {
        double maxabs = std::max(std::abs(res_seq[i]),std::abs(res[i]));
        if (maxabs == 0.0) maxabs=1.0;
        reldiff = std::max(reldiff, std::abs(res_seq[i] - res[i])/maxabs);
        diff = std::max(diff, std::abs(res_seq[i] - res[i]));
    }
    std::cout << "Max diff = " << diff << "  Max rel diff = " << reldiff << std::endl;

    /* Print out the result in a file */
    fprintf(fpt,"%s, %ld, %lg, %d\n", mode, elapsed_time, gpuflops, passed);
    fflush(fpt);

    free(res);
    free(res_seq);    
    

    return 0;
}