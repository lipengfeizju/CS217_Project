// Code Credit: Pengfei Li
// Email: pli081@ucr.edu
// All rights reserved

#include <iostream>
#include <numeric>
#include <cmath>
#include "icp.h"
#include "Eigen/Eigen"
#include <assert.h>
#include <iomanip>
#include <unistd.h>

#define BLOCK_SIZE 32
#define GRID_SIZE 2

#include <cublas_v2.h>
#include <cusolverDn.h>

#include "support.cu"

// #define NN_OPTIMIZE 0 

/***************************      Device Function           ********************************/

// Calculate distance in GPU
__device__ double dist_GPU(float x1, float y1, float z1, 
                         float x2, float y2, float z2){
    //dist = sqrt(pow(point1[0] - point2[0], 2) + pow(point1[1] - point2[1], 2) + pow(point1[2] - point2[2], 2));
    return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2) + pow(z1 - z2, 2));
}

/*****************************      Kernel Function         ******************************/
// Kernal function to find the nearest neighbor 
__global__ void nearest_neighbor_kernel(const float * src, const float * dst, int src_count, int dst_count, int *best_neighbor, double *best_dist){
    // Kernal function to find the nearest neighbor 
    // src: source point cloud array, (num_pts, 3), stored in ColMajor (similar for dst)
    // best_neighbor: best neigbor index in dst point set
    // best_dist    : best neigbor distance from src to dst

    // Dynamic reserve shared mem
    extern __shared__ float shared_mem[]; 

    int num_dst_pts_per_thread = (dst_count - 1)/(gridDim.x * blockDim.x) + 1;
    int num_src_pts_per_thread = (src_count - 1)/(gridDim.x * blockDim.x) + 1;

    int num_dst_pts_per_block = num_dst_pts_per_thread * blockDim.x;
    int num_src_pts_per_block = num_src_pts_per_thread * blockDim.x;

    float *shared_points = (float *)shared_mem;                // num_dst_pts_per_thread * blockDim.x * 3         
    

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
       float x1, y1, z1;
       float x2, y2, z2;
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

// Kernal function to find the nearest neighbor 
__global__ void nearest_neighbor_naive_kernel(const float * src, const float * dst, int src_count, int dst_count, int *best_neighbor, double *best_dist){
    // Kernal function to find the nearest neighbor 
    // src: source point cloud array, (num_pts, 3), stored in ColMajor (similar for dst)
    // best_neighbor: best neigbor index in dst point set
    // best_dist    : best neigbor distance from src to dst

    // Dynamic reserve shared mem
    int num_src_pts_per_thread = (src_count - 1)/(gridDim.x * blockDim.x) + 1;

    double current_best_dist = INF;
    int current_best_neighbor = 0;
    int current_index_src = 0;

    float x1, y1, z1;
    float x2, y2, z2;
    double dist;
    for(int j = 0; j < num_src_pts_per_thread; j++){
        current_index_src =  blockIdx.x * blockDim.x * num_src_pts_per_thread + j * blockDim.x + threadIdx.x;
        if (current_index_src < src_count){

            current_best_dist = INF;
            current_best_neighbor = 0;
            x1 = src[current_index_src];
            y1 = src[current_index_src + src_count];
            z1 = src[current_index_src + src_count*2];

            for(int current_index_dst = 0; current_index_dst < dst_count; current_index_dst++){
                x2 = dst[current_index_dst];
                y2 = dst[current_index_dst + dst_count];
                z2 = dst[current_index_dst + dst_count*2];
                dist = dist_GPU(x1, y1, z1, x2, y2, z2);
                if(dist < current_best_dist){
                    current_best_dist = dist;
                    current_best_neighbor = current_index_dst;
                }
            }

            best_dist[current_index_src] = current_best_dist;  //INF
            best_neighbor[current_index_src] = current_best_neighbor; //0
        }
    }
}

// Change point array order given the index array
__global__ void point_array_chorder(const float *src, const int *indices, int num_points, float *src_chorder){
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


/*******************************    Helper Function          ***************************/

__host__ void best_trasnform_SVD(cublasHandle_t handle, cusolverDnHandle_t solver_handle, const float *src_zm_device, const float *dst_chorder_zm_device, const float *sum_device_src, const float *sum_device_dst, int num_data_pts, double *trans_matrix_device){

    const float alf = 1;
    const float bet = 0;
    // const float *alpha = &alf;
    // const float *beta = &bet;


    /***********************            Calculate H matrix            **********************************/
    float *H_matrix_device;
    check_return_status(cudaMalloc((void**)&H_matrix_device, 3 * 3 * sizeof(float)));

    // src_zm_device(N,3) dst_chorder_zm_device(N,3)
    // src_zm_device.T  *  dst_chorder_zm_device
    // cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, K, M, &alpha, A, M, A, M, &beta, B, N);
    // A(MxN) K = N  A'(N,M)
    int m = 3, k = num_data_pts, n = 3;
    int lda=k, ldb=k, ldc=m;
    check_return_status(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alf, src_zm_device, lda, dst_chorder_zm_device, ldb, &bet, H_matrix_device, ldc));
    //print_matrix_device<<<1,1>>>(H_matrix_device, 3, 3);
    

    /****************************   SVD decomposition for trans_matrix   *****************************/
    // --- gesvd only supports Nrows >= Ncols
    // --- column major memory ordering

    const int Nrows = 3;
    const int Ncols = 3;

    // --- cuSOLVE input/output parameters/arrays
    int work_size = 0;
    int *devInfo;           check_return_status(cudaMalloc(&devInfo,          sizeof(int)));

    // --- Setting the device matrix and moving the host matrix to the device
    double *d_A;            check_return_status(cudaMalloc(&d_A,      Nrows * Ncols * sizeof(double)));
    cast_float_to_double<<<1,1>>>(H_matrix_device, d_A, Nrows * Ncols);

    
    // --- device side SVD workspace and matrices
    double *d_U;            check_return_status(cudaMalloc(&d_U,  Nrows * Nrows     * sizeof(double)));
    double *d_Vt;            check_return_status(cudaMalloc(&d_Vt,  Ncols * Ncols     * sizeof(double)));
    double *d_S;            check_return_status(cudaMalloc(&d_S,  min(Nrows, Ncols) * sizeof(double)));

    // --- CUDA SVD initialization
    check_return_status(cusolverDnDgesvd_bufferSize(solver_handle, Nrows, Ncols, &work_size));
    double *work;   check_return_status(cudaMalloc(&work, work_size * sizeof(double)));

    // --- CUDA SVD execution
    check_return_status(cusolverDnDgesvd(solver_handle, 'A', 'A', Nrows, Ncols, d_A, Nrows, d_S, d_U, Nrows, d_Vt, Ncols, work, work_size, NULL, devInfo));
    int devInfo_h = 0;  check_return_status(cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
    if (devInfo_h != 0) std::cout   << "Unsuccessful SVD execution\n\n";
    check_return_status(cudaFree(work));
    check_return_status(cudaFree(devInfo));

    check_return_status(cudaFree(H_matrix_device));
    check_return_status(cudaFree(d_A));
    check_return_status(cudaFree(d_S));

    /**************************      calculating rotation matrix       ******************************/
    const double alfd = 1;
    const double betd = 0;
    const double *alphad = &alfd;
    const double *betad = &betd;

    double *rot_matrix_device;
    check_return_status(cudaMalloc((void**)&rot_matrix_device, 3 * 3 * sizeof(double)));

    m = 3; k = 3; n = 3;
    lda=k; ldb=k; ldc=m;
    // Vt.transpose()*U.transpose();
    check_return_status(cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, alphad, d_Vt, lda, d_U, ldb, betad, rot_matrix_device, ldc));
    check_return_status(cudaFree(d_Vt));
    check_return_status(cudaFree(d_U));

    /***************************      calculating translation matrix    ******************************/
    double *t_matrix_device;
    check_return_status(cudaMalloc((void**)&t_matrix_device, 3 * sizeof(double)));

    m = 3; k = 3; n = 1; //(m,k), (k,n)  -> (m, n)
    lda=m; ldb=k; ldc=m;
    double *sum_device_src_d;            check_return_status(cudaMalloc(&sum_device_src_d, 3 * sizeof(double)));
    cast_float_to_double<<<1,1>>>(sum_device_src, sum_device_src_d, 3);
    check_return_status(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alphad, rot_matrix_device, lda, sum_device_src_d, ldb, betad, t_matrix_device, ldc));
    check_return_status(cudaFree(sum_device_src_d));

    const double scale_trans = -1;
    check_return_status(cublasDscal(handle, 3, &scale_trans, t_matrix_device, 1));

    double *sum_device_dst_d;            check_return_status(cudaMalloc(&sum_device_dst_d, 3 * sizeof(double)));
    cast_float_to_double<<<1,1>>>(sum_device_dst, sum_device_dst_d, 3);
    const double scale_trans_1 = 1;
    check_return_status(cublasDaxpy(handle, 3, &scale_trans_1, sum_device_dst_d, 1, t_matrix_device, 1));
    check_return_status(cudaFree(sum_device_dst_d));

    const double avg_trans = 1/(1.0*num_data_pts);
    check_return_status(cublasDscal(handle, 3, &avg_trans, t_matrix_device, 1));

    /*************         final transformation         ********************/
    // Set the last value to one
    double temp_one = 1;
    check_return_status(cublasSetVector(1, sizeof(double), &temp_one, 1, trans_matrix_device + 15, 1));
    for( int i = 0; i< 3; i++){
        check_return_status(cublasDcopy(handle, 3, rot_matrix_device + i * 3, 1, trans_matrix_device + i * 4, 1));
    }
    check_return_status(cublasDcopy(handle, 3, t_matrix_device, 1, trans_matrix_device + 12, 1));
    check_return_status(cudaFree(rot_matrix_device));
    check_return_status(cudaFree(t_matrix_device));

}

__host__ void zero_center_points(cublasHandle_t handle, const float *point_array_device, const float *ones_device, int num_data_pts, float *point_array_zm_device, float *sum_device_dst){

    const float alf = 1;
    const float bet = 0;
    // const float *alpha = &alf;
    // const float *beta = &bet;

    float *average_host = (float *)malloc(3*sizeof(float));

    /*******************************  zero center dst point array    *****************************************/
    // Do the actual multiplication 
    // op ( A ) m × k , op ( B ) k × n and C m × n ,
    // cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    int m = 1, k = num_data_pts, n = 3;
    int lda=m,ldb=k,ldc=m;
    check_return_status(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alf, ones_device, lda, point_array_device, ldb, &bet, sum_device_dst, ldc));
    
    cublasGetVector(3, sizeof(float), sum_device_dst, 1, average_host, 1);
    
    for(int i = 0; i < 3; i++)  average_host[i] /= num_data_pts;
    
    
    check_return_status(cudaMemcpy(point_array_zm_device, point_array_device, 3 * num_data_pts * sizeof(float), cudaMemcpyDeviceToDevice));
    
    for(int i = 0; i < 3; i++)
    {
        const float avg = -average_host[i];
        check_return_status(cublasSaxpy(handle, num_data_pts, &avg, ones_device, 1, point_array_zm_device + i*num_data_pts, 1));
    }

}


/****************************     Warper function *******************************/
// To simplify the function, warper function assumes every device variable is correctly allocated and initialized
// Don't use this unless you are certain about that

__host__ void _nearest_neighbor_cuda_warper(const float *src_device, const float *dst_device, int row_src, int row_dst, double *best_dist_device, int *best_neighbor_device){

    int num_dst_pts_per_thread = (row_dst - 1)/(GRID_SIZE * BLOCK_SIZE) + 1;
    
    int dyn_size_1 = num_dst_pts_per_thread * BLOCK_SIZE * 3 * sizeof(float);  // memory reserved for shared_points

#ifndef NN_OPTIMIZE
    nearest_neighbor_naive_kernel<<<GRID_SIZE, BLOCK_SIZE >>>(src_device, dst_device, row_src, row_dst, best_neighbor_device, best_dist_device);
#elif NN_OPTIMIZE == 0
    nearest_neighbor_kernel<<<GRID_SIZE, BLOCK_SIZE, (dyn_size_1) >>>(src_device, dst_device, row_src, row_dst, best_neighbor_device, best_dist_device);
#elif NN_OPTIMIZE == 1
    dim3 fullGrids((row_src + BLOCK_SIZE - 1) / BLOCK_SIZE);
    nearest_neighbor_naive_kernel<<<fullGrids, BLOCK_SIZE >>>(src_device, dst_device, row_src, row_dst, best_neighbor_device, best_dist_device);
#endif
}

__host__ void _apply_optimal_transform_cuda_warper(cublasHandle_t handle, cusolverDnHandle_t solver_handle, const float *dst_device, const float *src_device, const int *neighbor_device, const float *ones_device, int num_data_pts,
        float *dst_chorder_device, float *dst_chorder_zm_device, float *src_zm_device, float *sum_device_dst, float *sum_device_src,
        float *src_4d_t_device, float *src_4d_device
    ){

    /*****************************   change order based on the nearest neighbor ******************************/
    point_array_chorder<<<GRID_SIZE, BLOCK_SIZE>>>(dst_device, neighbor_device, num_data_pts, dst_chorder_device);
        
    /******************************   Calculate Transformation with SVD    ************************************/
    zero_center_points(handle, dst_chorder_device, ones_device, num_data_pts, dst_chorder_zm_device, sum_device_dst);
    zero_center_points(handle, src_device, ones_device, num_data_pts, src_zm_device, sum_device_src);
    
    double *trans_matrix_device; //matrix size is (4,4)
    check_return_status(cudaMalloc((void**)&trans_matrix_device, 4 * 4 * sizeof(double)));
    
    best_trasnform_SVD(handle, solver_handle, src_zm_device, dst_chorder_zm_device, sum_device_src, sum_device_dst, num_data_pts, trans_matrix_device);
    

    /********************************       Apply transformation       **************************************/
    // Convert to float data
    float *trans_matrix_f_device; //matrix size is (4,4)
    check_return_status(cudaMalloc((void**)&trans_matrix_f_device, 4 * 4 * sizeof(float)));
    cast_double_to_float<<<1,1>>>(trans_matrix_device, trans_matrix_f_device, 16);

    // Matrix multiplication
    const float alf = 1;
    const float bet = 0;
    int m = 4, k = 4, n = num_data_pts;
    int lda=m,ldb=n,ldc=m;
    check_return_status(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alf, trans_matrix_f_device, lda, src_4d_device, ldb, &bet, src_4d_t_device, ldc));

    /*******************************      Transpose the matrix       *****************************************/
    m = num_data_pts; n = 4;
    lda=n,ldb=n,ldc=m;
    check_return_status(cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n,
        &alf, src_4d_t_device, lda,
        &bet, src_4d_t_device, ldb,
        src_4d_device, ldc));


    
    check_return_status(cudaFree(trans_matrix_device));
    check_return_status(cudaFree(trans_matrix_f_device));
    
}

/***************************       Main  algorithm *********************************/
__host__ int icp_cuda(const Eigen::MatrixXf &dst,  const Eigen::MatrixXf &src, int max_iterations, float tolerance, Eigen::MatrixXf &src_transformed, NEIGHBOR &neighbor_out){
    assert(src_transformed.cols() == 4 && src_transformed.rows() == src.rows());
    assert(src.rows() == dst.rows());// && dst.rows() == dst_chorder.rows());
    assert(src.cols() == dst.cols());// && dst.cols() == dst_chorder.cols());
    assert(dst.cols() == 3);
    //assert(dst.rows() == neighbor.indices.size());

    // Host variables declaration
    int num_data_pts = dst.rows();
    const float *dst_host         = dst.data();
    const float *src_host         = src.data();
    float *gpu_temp_res          = src_transformed.data();
    int *best_neighbor_host = (int *)malloc(num_data_pts*sizeof(int)); 
    double *best_dist_host  = (double *)malloc(num_data_pts*sizeof(double));

    // Device variables declaration
    float *dst_chorder_device, *dst_device, *src_device, *src_4d_device;
    float *src_4d_t_device; // temp result
    float *dst_chorder_zm_device, *src_zm_device;
    int *neighbor_device;

    //int *best_neighbor_device;
    double *best_dist_device;

    // CUBLAS and CUSOLVER initialization
    // Create a handle for CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    // CUDA solver initialization
    cusolverDnHandle_t solver_handle;
    cusolverDnCreate(&solver_handle);

    float *ones_host = (float *)malloc(num_data_pts*sizeof(float));
    for(int i = 0; i< num_data_pts; i++){
        ones_host[i] = 1;}
    float *average_host = (float *)malloc(3*sizeof(float));
    float *ones_device, *sum_device_src, *sum_device_dst;
    
    /*************************       CUDA memory operations          ********************************/
    // Initialize the CUDA memory
    check_return_status(cudaMalloc((void**)&dst_device         , 3 * num_data_pts * sizeof(float)));
    check_return_status(cudaMalloc((void**)&dst_chorder_device , 3 * num_data_pts * sizeof(float)));
    check_return_status(cudaMalloc((void**)&src_device         , 3 * num_data_pts * sizeof(float)));
    check_return_status(cudaMalloc((void**)&src_4d_device      , 4 * num_data_pts * sizeof(float)));
    check_return_status(cudaMalloc((void**)&src_4d_t_device, 4 * num_data_pts * sizeof(float)));

    check_return_status(cudaMalloc((void**)&neighbor_device   , num_data_pts * sizeof(int)));
    check_return_status(cudaMalloc((void**)&dst_chorder_zm_device, 3 * num_data_pts * sizeof(float)));
    check_return_status(cudaMalloc((void**)&src_zm_device, 3 * num_data_pts * sizeof(float)));

    check_return_status(cudaMalloc((void**)&ones_device, num_data_pts * sizeof(float)));
    check_return_status(cudaMalloc((void**)&sum_device_src, 3 * sizeof(float)));
    check_return_status(cudaMalloc((void**)&sum_device_dst, 3 * sizeof(float)));
    
    //check_return_status(cudaMalloc((void**)&best_neighbor_device, num_data_pts * sizeof(int)));
    check_return_status(cudaMalloc((void**)&best_dist_device, num_data_pts * sizeof(double)));

    // Copy data from host to device
    check_return_status(cudaMemcpy(dst_device, dst_host, 3 * num_data_pts * sizeof(float), cudaMemcpyHostToDevice));
    check_return_status(cudaMemcpy(src_device, src_host, 3 * num_data_pts * sizeof(float), cudaMemcpyHostToDevice));
    //check_return_status(cudaMemcpy(neighbor_device, &(neighbor.indices[0]),  num_data_pts * sizeof(int), cudaMemcpyHostToDevice));
    
    check_return_status(cublasSetVector(num_data_pts, sizeof(float), ones_host, 1, ones_device, 1));
    check_return_status(cudaMemcpy(src_4d_device, src_device, 3 * num_data_pts * sizeof(float), cudaMemcpyDeviceToDevice));
    check_return_status(cudaMemcpy(src_4d_device + 3 * num_data_pts, 
                                   ones_device, num_data_pts * sizeof(float), cudaMemcpyDeviceToDevice));
    
    /*******************************    Actual work done here                   ********************************/
    double prev_error = 0;
    double mean_error = 0;

    _nearest_neighbor_cuda_warper(src_4d_device, dst_device, num_data_pts, num_data_pts, best_dist_device, neighbor_device);
    check_return_status(cublasDasum(handle, num_data_pts, best_dist_device, 1, &prev_error));
    prev_error /= num_data_pts;

    //float tolerance = 1e-6;
    int iter = 0;
    for(int i = 0; i <max_iterations; i++){
    	//sleep(1);
        _apply_optimal_transform_cuda_warper(handle, solver_handle, dst_device, src_device, neighbor_device, ones_device, num_data_pts, //const input
            dst_chorder_device, dst_chorder_zm_device, src_zm_device, sum_device_dst, sum_device_src, // temp cache only
            src_4d_t_device, src_4d_device // results we care
        );
        //src_4d_device stored in col major, shape is (num_pts, 3)
        _nearest_neighbor_cuda_warper(src_4d_device, dst_device, num_data_pts, num_data_pts, best_dist_device, neighbor_device);
        
        
        check_return_status(cudaMemcpy(src_device, src_4d_device, 3* num_data_pts * sizeof(float), cudaMemcpyDeviceToDevice));
        
        check_return_status(cublasDasum(handle, num_data_pts, best_dist_device, 1, &mean_error));
        mean_error /= num_data_pts;
        std::cout << mean_error  << std::endl;
        if (abs(prev_error - mean_error) < tolerance){
            break;
        }
        // Calculate mean error and compare with previous error
        
        prev_error = mean_error;
        iter = i + 2;
    }
    
    check_return_status(cudaMemcpy(best_neighbor_host, neighbor_device, num_data_pts * sizeof(int), cudaMemcpyDeviceToHost));
    check_return_status(cudaMemcpy(best_dist_host    , best_dist_device    , num_data_pts * sizeof(double), cudaMemcpyDeviceToHost));

    neighbor_out.distances.clear(); 
    neighbor_out.indices.clear();
    for(int i = 0; i < num_data_pts; i++){
        neighbor_out.distances.push_back(best_dist_host[i]);
        neighbor_out.indices.push_back(best_neighbor_host[i]);
    }

    
    /**********************************  Final cleanup steps     ********************************************/
    // Destroy the handle
    cublasDestroy(handle);
    cusolverDnDestroy(solver_handle);

    // Final result copy back
    check_return_status(cudaMemcpy(gpu_temp_res, src_4d_device, 4 * num_data_pts * sizeof(float), cudaMemcpyDeviceToHost));
    // check_return_status(cudaMemcpy(gpu_temp_res, trans_matrix_device, 4 * 4 * sizeof(double), cudaMemcpyDeviceToHost));

    

    // Free all variables
    
    check_return_status(cudaFree(dst_device));
    check_return_status(cudaFree(src_device));
    check_return_status(cudaFree(dst_chorder_device));
    check_return_status(cudaFree(neighbor_device));
    check_return_status(cudaFree(dst_chorder_zm_device));
    check_return_status(cudaFree(src_zm_device));
    check_return_status(cudaFree(ones_device));
    
    return iter;
}

// Host function to prepare data
__host__ NEIGHBOR nearest_neighbor_cuda(const Eigen::MatrixXf &src, const Eigen::MatrixXf &dst){
    /*
    src : src point cloud matrix with size (num_point, 3)
    dst : dst point cloud matrix with size (num_point, 3)
    the matrix is stored in ColMajor by default
    */

    NEIGHBOR neigh;
    
    int row_src = src.rows();
    int row_dst = dst.rows();
    assert(row_src == row_src);
    
    //Initialize Host variables
    const float *src_host = src.data();
    const float *dst_host = dst.data();
    int *best_neighbor_host = (int *)malloc(row_src*sizeof(int)); 
    double *best_dist_host  = (double *)malloc(row_src*sizeof(double));

    // Initialize Device variables
    float *src_device, *dst_device;
    int *best_neighbor_device;
    double *best_dist_device;

    check_return_status(cudaMalloc((void**)&src_device, 3 * row_src * sizeof(float)));
    check_return_status(cudaMalloc((void**)&dst_device, 3 * row_dst * sizeof(float)));
    check_return_status(cudaMalloc((void**)&best_neighbor_device, row_src * sizeof(int)));
    check_return_status(cudaMalloc((void**)&best_dist_device, row_src * sizeof(double)));

    check_return_status(cudaMemcpy(src_device, src_host, 3 * row_src * sizeof(float), cudaMemcpyHostToDevice));
    check_return_status(cudaMemcpy(dst_device, dst_host, 3 * row_dst * sizeof(float), cudaMemcpyHostToDevice));

    _nearest_neighbor_cuda_warper(src_device, dst_device, row_src, row_dst, best_dist_device, best_neighbor_device);
    
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

/*************************************************************************************************/
/****************************** Single step functions for DEBUG ***********************************/
/*************************************************************************************************/

__host__ double apply_optimal_transform_cuda(const Eigen::MatrixXf &dst,  const Eigen::MatrixXf &src, Eigen::MatrixXf &src_transformed, const NEIGHBOR &neighbor){

    assert(src_transformed.cols() == 4 && src_transformed.rows() == src.rows());
    assert(src.rows() == dst.rows());// && dst.rows() == dst_chorder.rows());
    assert(src.cols() == dst.cols());// && dst.cols() == dst_chorder.cols());
    assert(dst.cols() == 3);
    assert(dst.rows() == neighbor.indices.size());

    // Host variables declaration
    const float *dst_host         = dst.data();
    const float *src_host         = src.data();
    float *gpu_temp_res          = src_transformed.data();
    int num_data_pts = dst.rows();

    // Device variables declaration
    float *dst_chorder_device, *dst_device, *src_device, *src_4d_device;
    float *src_4d_t_device; // temp result
    float *dst_chorder_zm_device, *src_zm_device;
    int *neighbor_device;

    // CUBLAS and CUSOLVER initialization
    // Create a handle for CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    // CUDA solver initialization
    cusolverDnHandle_t solver_handle;
    cusolverDnCreate(&solver_handle);

    float *ones_host = (float *)malloc(num_data_pts*sizeof(float));
    for(int i = 0; i< num_data_pts; i++){
        ones_host[i] = 1;}
    float *average_host = (float *)malloc(3*sizeof(float));
    float *ones_device, *sum_device_src, *sum_device_dst;
    
    /*************************       CUDA memory operations          ********************************/
    // Initialize the CUDA memory
    check_return_status(cudaMalloc((void**)&dst_device         , 3 * num_data_pts * sizeof(float)));
    check_return_status(cudaMalloc((void**)&dst_chorder_device , 3 * num_data_pts * sizeof(float)));
    check_return_status(cudaMalloc((void**)&src_device         , 3 * num_data_pts * sizeof(float)));
    check_return_status(cudaMalloc((void**)&src_4d_device      , 4 * num_data_pts * sizeof(float)));
    check_return_status(cudaMalloc((void**)&src_4d_t_device, 4 * num_data_pts * sizeof(float)));

    check_return_status(cudaMalloc((void**)&neighbor_device   , num_data_pts * sizeof(int)));
    check_return_status(cudaMalloc((void**)&dst_chorder_zm_device, 3 * num_data_pts * sizeof(float)));
    check_return_status(cudaMalloc((void**)&src_zm_device, 3 * num_data_pts * sizeof(float)));

    check_return_status(cudaMalloc((void**)&ones_device, num_data_pts * sizeof(float)));
    check_return_status(cudaMalloc((void**)&sum_device_src, 3 * sizeof(float)));
    check_return_status(cudaMalloc((void**)&sum_device_dst, 3 * sizeof(float)));
    

    // Copy data from host to device
    check_return_status(cudaMemcpy(dst_device, dst_host, 3 * num_data_pts * sizeof(float), cudaMemcpyHostToDevice));
    check_return_status(cudaMemcpy(src_device, src_host, 3 * num_data_pts * sizeof(float), cudaMemcpyHostToDevice));
    check_return_status(cudaMemcpy(neighbor_device, &(neighbor.indices[0]),  num_data_pts * sizeof(int), cudaMemcpyHostToDevice));
    
    check_return_status(cublasSetVector(num_data_pts, sizeof(float), ones_host, 1, ones_device, 1));
    check_return_status(cudaMemcpy(src_4d_device, src_device, 3 * num_data_pts * sizeof(float), cudaMemcpyDeviceToDevice));
    check_return_status(cudaMemcpy(src_4d_device + 3 * num_data_pts, 
                                   ones_device, num_data_pts * sizeof(float), cudaMemcpyDeviceToDevice));
    
    
    _apply_optimal_transform_cuda_warper(handle, solver_handle, dst_device, src_device, neighbor_device, ones_device, num_data_pts, //const input
        dst_chorder_device, dst_chorder_zm_device, src_zm_device, sum_device_dst, sum_device_src, // temp cache only
        src_4d_t_device, src_4d_device // results we care
    );
    
    /**********************************  Final cleanup steps     ********************************************/
    // Destroy the handle
    cublasDestroy(handle);
    cusolverDnDestroy(solver_handle);

    // Final result copy back
    check_return_status(cudaMemcpy(gpu_temp_res, src_4d_device, 4 * num_data_pts * sizeof(float), cudaMemcpyDeviceToHost));
    // check_return_status(cudaMemcpy(gpu_temp_res, trans_matrix_device, 4 * 4 * sizeof(double), cudaMemcpyDeviceToHost));

    

    // Free all variables
    
    check_return_status(cudaFree(dst_device));
    check_return_status(cudaFree(src_device));
    check_return_status(cudaFree(dst_chorder_device));
    check_return_status(cudaFree(neighbor_device));
    check_return_status(cudaFree(dst_chorder_zm_device));
    check_return_status(cudaFree(src_zm_device));
    check_return_status(cudaFree(ones_device));
    
    

    return 0;
}

__host__ double single_step_ICP(const Eigen::MatrixXf &dst,  const Eigen::MatrixXf &src, const NEIGHBOR &neighbor, Eigen::MatrixXf &src_transformed, NEIGHBOR &neighbor_out){
    assert(src_transformed.cols() == 4 && src_transformed.rows() == src.rows());
    assert(src.rows() == dst.rows());// && dst.rows() == dst_chorder.rows());
    assert(src.cols() == dst.cols());// && dst.cols() == dst_chorder.cols());
    assert(dst.cols() == 3);
    assert(dst.rows() == neighbor.indices.size());

    // Host variables declaration
    int num_data_pts = dst.rows();
    const float *dst_host         = dst.data();
    const float *src_host         = src.data();
    float *gpu_temp_res          = src_transformed.data();
    int *best_neighbor_host = (int *)malloc(num_data_pts*sizeof(int)); 
    double *best_dist_host  = (double *)malloc(num_data_pts*sizeof(double));

    // Device variables declaration
    float *dst_chorder_device, *dst_device, *src_device, *src_4d_device;
    float *src_4d_t_device; // temp result
    float *dst_chorder_zm_device, *src_zm_device;
    int *neighbor_device;

    int *best_neighbor_device;
    double *best_dist_device;

    // CUBLAS and CUSOLVER initialization
    // Create a handle for CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    // CUDA solver initialization
    cusolverDnHandle_t solver_handle;
    cusolverDnCreate(&solver_handle);

    float *ones_host = (float *)malloc(num_data_pts*sizeof(float));
    for(int i = 0; i< num_data_pts; i++){
        ones_host[i] = 1;}
    float *average_host = (float *)malloc(3*sizeof(float));
    float *ones_device, *sum_device_src, *sum_device_dst;
    
    /*************************       CUDA memory operations          ********************************/
    // Initialize the CUDA memory
    check_return_status(cudaMalloc((void**)&dst_device         , 3 * num_data_pts * sizeof(float)));
    check_return_status(cudaMalloc((void**)&dst_chorder_device , 3 * num_data_pts * sizeof(float)));
    check_return_status(cudaMalloc((void**)&src_device         , 3 * num_data_pts * sizeof(float)));
    check_return_status(cudaMalloc((void**)&src_4d_device      , 4 * num_data_pts * sizeof(float)));
    check_return_status(cudaMalloc((void**)&src_4d_t_device, 4 * num_data_pts * sizeof(float)));

    check_return_status(cudaMalloc((void**)&neighbor_device   , num_data_pts * sizeof(int)));
    check_return_status(cudaMalloc((void**)&dst_chorder_zm_device, 3 * num_data_pts * sizeof(float)));
    check_return_status(cudaMalloc((void**)&src_zm_device, 3 * num_data_pts * sizeof(float)));

    check_return_status(cudaMalloc((void**)&ones_device, num_data_pts * sizeof(float)));
    check_return_status(cudaMalloc((void**)&sum_device_src, 3 * sizeof(float)));
    check_return_status(cudaMalloc((void**)&sum_device_dst, 3 * sizeof(float)));
    
    check_return_status(cudaMalloc((void**)&best_neighbor_device, num_data_pts * sizeof(int)));
    check_return_status(cudaMalloc((void**)&best_dist_device, num_data_pts * sizeof(double)));

    // Copy data from host to device
    check_return_status(cudaMemcpy(dst_device, dst_host, 3 * num_data_pts * sizeof(float), cudaMemcpyHostToDevice));
    check_return_status(cudaMemcpy(src_device, src_host, 3 * num_data_pts * sizeof(float), cudaMemcpyHostToDevice));
    check_return_status(cudaMemcpy(neighbor_device, &(neighbor.indices[0]),  num_data_pts * sizeof(int), cudaMemcpyHostToDevice));
    
    check_return_status(cublasSetVector(num_data_pts, sizeof(float), ones_host, 1, ones_device, 1));
    check_return_status(cudaMemcpy(src_4d_device, src_device, 3 * num_data_pts * sizeof(float), cudaMemcpyDeviceToDevice));
    check_return_status(cudaMemcpy(src_4d_device + 3 * num_data_pts, 
                                   ones_device, num_data_pts * sizeof(float), cudaMemcpyDeviceToDevice));
    
    /*******************************    Actual work done here                   ********************************/
    _apply_optimal_transform_cuda_warper(handle, solver_handle, dst_device, src_device, neighbor_device, ones_device, num_data_pts, //const input
        dst_chorder_device, dst_chorder_zm_device, src_zm_device, sum_device_dst, sum_device_src, // temp cache only
        src_4d_t_device, src_4d_device // results we care
    );
    //src_4d_device stored in col major, shape is (num_pts, 3)
    _nearest_neighbor_cuda_warper(src_4d_device, dst_device, num_data_pts, num_data_pts, best_dist_device, best_neighbor_device);
    check_return_status(cudaMemcpy(best_neighbor_host, best_neighbor_device, num_data_pts * sizeof(int), cudaMemcpyDeviceToHost));
    check_return_status(cudaMemcpy(best_dist_host    , best_dist_device    , num_data_pts * sizeof(double), cudaMemcpyDeviceToHost));
    
    double mean_error = 0;
    check_return_status(cublasDasum(handle, num_data_pts, best_dist_device, 1, &mean_error));

    neighbor_out.distances.clear(); 
    neighbor_out.indices.clear();
    for(int i = 0; i < num_data_pts; i++){
        neighbor_out.distances.push_back(best_dist_host[i]);
        neighbor_out.indices.push_back(best_neighbor_host[i]);
    }

    
    /**********************************  Final cleanup steps     ********************************************/
    // Destroy the handle
    cublasDestroy(handle);
    cusolverDnDestroy(solver_handle);

    // Final result copy back
    check_return_status(cudaMemcpy(gpu_temp_res, src_4d_device, 4 * num_data_pts * sizeof(float), cudaMemcpyDeviceToHost));
    // check_return_status(cudaMemcpy(gpu_temp_res, trans_matrix_device, 4 * 4 * sizeof(double), cudaMemcpyDeviceToHost));

    

    // Free all variables
    
    check_return_status(cudaFree(dst_device));
    check_return_status(cudaFree(src_device));
    check_return_status(cudaFree(dst_chorder_device));
    check_return_status(cudaFree(neighbor_device));
    check_return_status(cudaFree(dst_chorder_zm_device));
    check_return_status(cudaFree(src_zm_device));
    check_return_status(cudaFree(ones_device));
    
    return mean_error/num_data_pts;
}
