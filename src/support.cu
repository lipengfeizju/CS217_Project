// Copyright: Pengfei Li
// Email: pli081@ucr.edu
#include <iostream>

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