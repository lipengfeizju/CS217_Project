// Code Credit: Pengfei Li
// Email: pli081@ucr.edu

#include <iostream>
#include <string>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <stdio.h>

__global__ void print_matrix_device(const float *A, int nr_rows_A, int nr_cols_A) {
 
    for(int i = 0; i < nr_rows_A; ++i){
        for(int j = 0; j < nr_cols_A; ++j){
           printf("%f ", A[j * nr_rows_A + i]) ;
        }
        printf("\n");
   }
   printf("\n");
}
__host__ void print_matrix_host(const float *A, int nr_rows_A, int nr_cols_A) {

   for(int i = 0; i < nr_rows_A; ++i){
       for(int j = 0; j < nr_cols_A; ++j){
          printf("%f ", A[j * nr_rows_A + i]) ;
       }
       printf("\n");
  }
  printf("\n");
}

__global__ void cast_float_to_double(const float *src, double *dst, int num_points){
    int num_point_per_thread = (num_points - 1)/(gridDim.x * blockDim.x) + 1;
    int current_index = 0;
    for(int j = 0; j < num_point_per_thread; j++){
        current_index =  blockIdx.x * blockDim.x * num_point_per_thread + j * blockDim.x + threadIdx.x;
        if (current_index < num_points) dst[current_index] = (double)src[current_index];
    }
}
__global__ void cast_double_to_float(const double *src, float *dst, int num_points){
    int num_point_per_thread = (num_points - 1)/(gridDim.x * blockDim.x) + 1;
    int current_index = 0;
    for(int j = 0; j < num_point_per_thread; j++){
        current_index =  blockIdx.x * blockDim.x * num_point_per_thread + j * blockDim.x + threadIdx.x;
        if (current_index < num_points) dst[current_index] = (float)src[current_index];
    }
}

void check_return_status(cudaError_t cuda_ret){
    if(cuda_ret != cudaSuccess){
        std::cout << cuda_ret << std::endl;
        std::cout << "CUDA RETRUNED ERROR! Unable to allocate device memory"<< std::endl;
        exit(-1);
    }
}

void safe_exit(){
    std::cout << "So far so good, keep going !\n";
    exit(0);
}

void check_return_status(cublasStatus_t cublas_ret){
    if(cublas_ret != CUBLAS_STATUS_SUCCESS){
        std::cout << cublas_ret << std::endl;
        std::cout << "CUBLAS Returned error! please check"<< std::endl;
        exit(-1);
    }
}
void check_return_status(cusolverStatus_t error)
{
    if (error != CUSOLVER_STATUS_SUCCESS){
        std::string error_message;
        switch (error)
        {
            case CUSOLVER_STATUS_SUCCESS:
                error_message = "CUSOLVER_SUCCESS";

            case CUSOLVER_STATUS_NOT_INITIALIZED:
                error_message = "CUSOLVER_STATUS_NOT_INITIALIZED";

            case CUSOLVER_STATUS_ALLOC_FAILED:
                error_message = "CUSOLVER_STATUS_ALLOC_FAILED";

            case CUSOLVER_STATUS_INVALID_VALUE:
                error_message = "CUSOLVER_STATUS_INVALID_VALUE";

            case CUSOLVER_STATUS_ARCH_MISMATCH:
                error_message = "CUSOLVER_STATUS_ARCH_MISMATCH";

            case CUSOLVER_STATUS_EXECUTION_FAILED:
                error_message = "CUSOLVER_STATUS_EXECUTION_FAILED";

            case CUSOLVER_STATUS_INTERNAL_ERROR:
                error_message = "CUSOLVER_STATUS_INTERNAL_ERROR";

            case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
                error_message = "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
        }
        error_message = "<unknown>";
        std::cout << "CUBLAS Returned error! please check:"<< std::endl;
        std::cout << error_message<< std::endl;
        exit(-1);
    }
    
}

