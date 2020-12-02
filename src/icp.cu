// Copyright: Pengfei Li
// Email: pli081@ucr.edu

#include <iostream>
#include <numeric>
#include <cmath>
#include "icp.h"
#include "Eigen/Eigen"
#include <assert.h>
#include "support.cu"

#define BLOCK_SIZE 16
#define GRID_SIZE 16

#include <cublas_v2.h>

__device__ double dist_GPU(double x1, double y1, double z1, 
                         double x2, double y2, double z2){
    //dist = sqrt(pow(point1[0] - point2[0], 2) + pow(point1[1] - point2[1], 2) + pow(point1[2] - point2[2], 2));
    return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2) + pow(z1 - z2, 2));
}

__global__ void nearest_neighbor_kernel(const double * src, const double * dst, int src_count, int dst_count, int *best_neighbor, double *best_dist){
    // Dynamic reserve shared mem
    extern __shared__ double shared_mem[]; 

    int num_dst_pts_per_thread = (dst_count - 1)/(gridDim.x * blockDim.x) + 1;
    int num_src_pts_per_thread = (src_count - 1)/(gridDim.x * blockDim.x) + 1;

    int num_dst_pts_per_block = num_dst_pts_per_thread * blockDim.x;
    int num_src_pts_per_block = num_src_pts_per_thread * blockDim.x;

    double *shared_points = (double *)shared_mem;                // num_dst_pts_per_thread * blockDim.x * 3
    // num_src_pts_per_thread * blockDim.x
    // double *min_dis       = (double *)(shared_mem + num_dst_pts_per_thread * blockDim.x * 3 * sizeof(double)); 
    // num_src_pts_per_thread * blockDim.x
    // int    *min_ind       = (int *)(min_dis + num_src_pts_per_thread * blockDim.x * sizeof(double));           
    

    int current_index_dst = 0, current_index_src = 0, current_index_shared = 0;
    
    //Step 0: Initialize variables
    
    for(int j = 0; j < num_src_pts_per_thread; j++){
        current_index_src =  blockIdx.x * blockDim.x * num_src_pts_per_thread + j * blockDim.x + threadIdx.x;
        if (current_index_src < src_count){
            best_dist[current_index_src] = INF;  //INF
            best_neighbor[current_index_src] = 0; //0
        }
            
    }
    //printf("test");
    __syncthreads();

    int num_data_chunk = (src_count - 1)/(num_src_pts_per_thread * blockDim.x) + 1;

    for(int i = 0; i < num_data_chunk; i++){
       //Step 1: Copy part of dst points to shared memory
       for(int j = 0; j < num_dst_pts_per_thread; j++){
            // Memory coalescing index
            current_index_dst = i * num_dst_pts_per_block + j * blockDim.x + threadIdx.x;  // TODO: index rotating
            if (current_index_dst < dst_count){
               //Copy 3d points to shared memory
               for(int k = 0; k<3; k++){
                   current_index_shared = j * blockDim.x + threadIdx.x;
                   shared_points[3*current_index_shared]     = dst[current_index_dst];               // copy dst x
                   shared_points[3*current_index_shared + 1] = dst[current_index_dst + dst_count]; // copy dst y
                   shared_points[3*current_index_shared + 2] = dst[current_index_dst + dst_count*2]; // copy dst z
               }
           }
       }

       __syncthreads();
       double x1, y1, z1;
       double x2, y2, z2;
       double dist;
       //Step 2: find closest point from src to dst shared
       for(int j = 0; j < num_src_pts_per_thread; j++){
           current_index_src = blockIdx.x * num_src_pts_per_block + j * blockDim.x + threadIdx.x;
           if(current_index_src < src_count){
               x1 = src[current_index_src];
               y1 = src[current_index_src + src_count];
               z1 = src[current_index_src + src_count*2];

            //    best_dist[current_index_src] = z1;
            //    best_neighbor[current_index_src] = 10;
               for(int k = 0; k < num_dst_pts_per_block; k++){
                   //current_index_shared = k;
                   x2 = shared_points[3*k];
                   y2 = shared_points[3*k + 1];
                   z2 = shared_points[3*k + 2];
                   dist = dist_GPU(x1, y1, z1, x2, y2, z2);
                   if(dist < best_dist[current_index_src]){
                       best_dist[current_index_src] = dist;
                       current_index_dst = i * blockDim.x * num_dst_pts_per_thread + k;
                       best_neighbor[current_index_src] = current_index_dst;
                   }
               }
           }

      }
   }
    
}

__host__ NEIGHBOR nearest_neighbor_cuda(const Eigen::MatrixXd &src, const Eigen::MatrixXd &dst){
    /*
    src : src point cloud matrix with size (num_point, 3)
    dst : dst point cloud matrix with size (num_point, 3)
    the matrix is stored in ColMajor by default
    */

    NEIGHBOR neigh;
    
    int row_src = src.rows();
    int row_dst = dst.rows();

    // const double * vc = dst.data();
    // for(int j = 0; j < 10; j++)
    //     std::cout << vc[j] << ", ";
    // std::cout << std::endl << std::endl;
    // std::cout << dst( Eigen::seqN(0,4), Eigen::all) << std::endl;
    
    //Initialize Host variables
    const double *src_host = src.data();
    const double *dst_host = dst.data();
    int *best_neighbor_host = (int *)malloc(row_src*sizeof(int)); 
    double *best_dist_host  = (double *)malloc(row_src*sizeof(double));

    // Initialize Device variables
    double *src_device, *dst_device;
    int *best_neighbor_device;
    double *best_dist_device;

    check_return_status(cudaMalloc((void**)&src_device, 3 * row_src * sizeof(double)));
    check_return_status(cudaMalloc((void**)&dst_device, 3 * row_dst * sizeof(double)));
    check_return_status(cudaMalloc((void**)&best_neighbor_device, row_src * sizeof(int)));
    check_return_status(cudaMalloc((void**)&best_dist_device, row_src * sizeof(double)));

    check_return_status(cudaMemcpy(src_device, src_host, 3 * row_src * sizeof(double), cudaMemcpyHostToDevice));
    check_return_status(cudaMemcpy(dst_device, dst_host, 3 * row_dst * sizeof(double), cudaMemcpyHostToDevice));

    int num_dst_pts_per_thread = (row_dst - 1)/(GRID_SIZE * BLOCK_SIZE) + 1;
    
    int dyn_size_1 = num_dst_pts_per_thread * BLOCK_SIZE * 3 * sizeof(double);  // memory reserved for shared_points

    nearest_neighbor_kernel<<<GRID_SIZE, BLOCK_SIZE, (dyn_size_1) >>>(src_device, dst_device, row_src, row_dst, best_neighbor_device, best_dist_device);
    
    

    check_return_status(cudaMemcpy(best_neighbor_host, best_neighbor_device, row_src * sizeof(int), cudaMemcpyDeviceToHost));
    check_return_status(cudaMemcpy(best_dist_host    , best_dist_device    , row_src * sizeof(double), cudaMemcpyDeviceToHost));
    
    for(int i = 0; i < row_src; i++){
        neigh.distances.push_back(best_dist_host[i]);
        neigh.indices.push_back(best_neighbor_host[i]);
    }
    
    free(best_neighbor_host);
    free(best_dist_host);
    cudaFree(src_device);
    cudaFree(dst_device);
    cudaFree(best_neighbor_device);
    cudaFree(best_dist_device);
    return neigh;
}

__global__ void point_array_chorder(const float *src, float *src_chorder, const int *indices, int num_points){
    int num_point_per_thread = (num_points - 1)/(gridDim.x * blockDim.x) + 1;
    int current_index = 0;
    int target_index = 0;

    for(int j = 0; j < num_point_per_thread; j++){
        current_index =  blockIdx.x * blockDim.x * num_point_per_thread + j * blockDim.x + threadIdx.x;
        if (current_index < num_points){
            target_index = indices[current_index];
            src_chorder[current_index]                 =  src[target_index];     //x
            src_chorder[current_index + num_points  ]  =  src[target_index + num_points  ];     //y
            src_chorder[current_index + num_points*2]  =  src[target_index + num_points*2];     //z
        }
    }
}

__host__ double cal_T_matrix_cuda(const Eigen::MatrixXf &dst,  const Eigen::MatrixXf &src, Eigen::MatrixXf &H_matrix, const NEIGHBOR &neighbor){

    assert(H_matrix.rows() == 3 && H_matrix.cols() == 3);
    assert(src.rows() == dst.rows());// && dst.rows() == dst_chorder.rows());
    assert(src.cols() == dst.cols());// && dst.cols() == dst_chorder.cols());
    assert(dst.cols() == 3);
    assert(dst.rows() == neighbor.indices.size());


    int num_data_pts = dst.rows();
    float *dst_chorder_device;
    float *dst_device, *src_device;
    int *neighbor_device;

    const float *dst_host         = dst.data();
    const float *src_host         = src.data();
    float *gpu_temp_res           = H_matrix.data();
    
    
    check_return_status(cudaMalloc((void**)&dst_device        , 3 * num_data_pts * sizeof(float)));
    check_return_status(cudaMalloc((void**)&src_device        , 3 * num_data_pts * sizeof(float)));
    check_return_status(cudaMalloc((void**)&dst_chorder_device, 3 * num_data_pts * sizeof(float)));
    check_return_status(cudaMalloc((void**)&neighbor_device   , num_data_pts * sizeof(int)));

    check_return_status(cudaMemcpy(dst_device, dst_host, 3 * num_data_pts * sizeof(float), cudaMemcpyHostToDevice));
    check_return_status(cudaMemcpy(src_device, src_host, 3 * num_data_pts * sizeof(float), cudaMemcpyHostToDevice));
    check_return_status(cudaMemcpy(neighbor_device, &(neighbor.indices[0]),  num_data_pts * sizeof(int), cudaMemcpyHostToDevice));
    

    point_array_chorder<<<GRID_SIZE, BLOCK_SIZE>>>(dst_device, dst_chorder_device, neighbor_device, num_data_pts);
    
    const float alf = 1;
    const float bet = 0;
    const float *alpha = &alf;
    const float *beta = &bet;


    // Matrix calculation
    // Create a handle for CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    
    float *ones_host = (float *)malloc(num_data_pts*sizeof(float));
    for(int i = 0; i< num_data_pts; i++){
        ones_host[i] = 1;}
    float *average_host = (float *)malloc(3*sizeof(float));
    float *ones_device, *average_device;
    
    check_return_status(cudaMalloc((void**)&ones_device, num_data_pts * sizeof(float)));
    check_return_status(cudaMalloc((void**)&average_device, 3 * sizeof(float)));
    check_return_status(cublasSetVector(num_data_pts, sizeof(float), ones_host, 1, ones_device, 1));
    
    /*******************************  zero center dst point array    *****************************************/
    // Do the actual multiplication 
    // op ( A ) m × k , op ( B ) k × n and C m × n ,
    // cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    int m = 1, k = num_data_pts, n = 3;
    int lda=m,ldb=k,ldc=m;
    check_return_status(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, ones_device, lda, dst_chorder_device, ldb, beta, average_device, ldc));
    //print_matrix_device<<<1,1>>>(dst_chorder_device, k, n);
    
    cublasGetVector(3, sizeof(float), average_device, 1, average_host, 1);
    
    for(int i = 0; i < 3; i++)  average_host[i] /= num_data_pts;
    // std::cout << average_host[0] << ", " << average_host[1] << ", "  << average_host[2] << std::endl;
    
    float *dst_chorder_zm_device;
    check_return_status(cudaMalloc((void**)&dst_chorder_zm_device, 3 * num_data_pts * sizeof(float)));
    check_return_status(cudaMemcpy(dst_chorder_zm_device, dst_chorder_device, 3 * num_data_pts * sizeof(float), cudaMemcpyDeviceToDevice));
    
    for(int i = 0; i < 3; i++)
    {
        const float avg = -average_host[i];
        check_return_status(cublasSaxpy(handle, num_data_pts, &avg, ones_device, 1, dst_chorder_zm_device + i*num_data_pts, 1));
    }

    /******************************   zero center dst point array    ************************************/
    m = 1, k = num_data_pts, n = 3;
    lda=m,ldb=k,ldc=m;
    check_return_status(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, ones_device, lda, src_device, ldb, beta, average_device, ldc));
    
    cublasGetVector(3, sizeof(float), average_device, 1, average_host, 1);
    for(int i = 0; i < 3; i++)  average_host[i] /= num_data_pts;
    // std::cout << average_host[0] << ", " << average_host[1] << ", "  << average_host[2] << std::endl;

    float *src_zm_device;
    check_return_status(cudaMalloc((void**)&src_zm_device, 3 * num_data_pts * sizeof(float)));
    check_return_status(cudaMemcpy(src_zm_device, src_device, 3 * num_data_pts * sizeof(float), cudaMemcpyDeviceToDevice));
    
    for(int i = 0; i < 3; i++)
    {
        const float avg = -average_host[i];
        check_return_status(cublasSaxpy(handle, num_data_pts, &avg, ones_device, 1, src_zm_device + i*num_data_pts, 1));
    }

    /*********************************************************/
    

    float *trans_matrix_device;
    check_return_status(cudaMalloc((void**)&trans_matrix_device, 3 * 3 * sizeof(float)));

    // src_zm_device(N,3) dst_chorder_zm_device(N,3)
    // src_zm_device.T  *  dst_chorder_zm_device

    //cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, K, M, &alpha, A, M, A, M, &beta, B, N);
    //A(MxN) K = N  A'(N,M)
    m = 3; k = num_data_pts; n = 3;
    lda=k; ldb=k; ldc=m;
    check_return_status(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, alpha, src_zm_device, lda, dst_chorder_zm_device, ldb, beta, trans_matrix_device, ldc));
    print_matrix_device<<<1,1>>>(trans_matrix_device, 3, 3);
    /*********************************************************/
    

    //check_return_status(cudaMemcpy(gpu_temp_res, src_zm_device, 3 * num_data_pts * sizeof(float), cudaMemcpyDeviceToHost));
    check_return_status(cudaMemcpy(gpu_temp_res, trans_matrix_device, 3 * 3 * sizeof(float), cudaMemcpyDeviceToHost));
    // Destroy the handle
    cublasDestroy(handle);

    return 0;
}