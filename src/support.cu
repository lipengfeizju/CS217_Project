// Copyright: Pengfei Li
// Email: pli081@ucr.edu
#include <iostream>
#include <cublas_v2.h>

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