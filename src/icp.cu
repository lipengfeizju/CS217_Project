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
        // int i = 0; 
        // i = blockIdx.x;
        // current_index_src =  i * blockDim.x * num_src_pts_per_thread + j * blockDim.x + threadIdx.x;
        current_index_src =  blockIdx.x * blockDim.x * num_src_pts_per_thread + j * blockDim.x + threadIdx.x;
        if (current_index_src < src_count){
            best_dist[current_index_src] = INF;  //INF
            best_neighbor[current_index_src] = 0; //0
        }
            
    }
    
    // for(int i = 0; i < gridDim.x; i++){
    //     //int i = 0;
    //     //for(int j = 0; j < num_src_pts_per_thread; j++){
    //         for(int k = 0; k < num_dst_pts_per_block; k++){
    //             current_index_dst = i * blockDim.x * num_dst_pts_per_thread + k;
    //             best_dist[current_index_dst] = INF; //0
    //         }
    //     //}
    // }
    
    __syncthreads();
//
//   //TODO : remove redundant loop
   for(int i = 0; i < gridDim.x; i++){
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
                       // TODO: index rotating for current_index_dst
                       //current_index_dst = i * blockDim.x * num_dst_pts_per_thread + k * num_dst_pts_per_thread + threadIdx.x;
                       ///TBD
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
    // int num_src_pts_per_thread = (row_src - 1)/(GRID_SIZE * BLOCK_SIZE) + 1;
    
    int dyn_size_1 = num_dst_pts_per_thread * BLOCK_SIZE * 3 * sizeof(double);  // memory reserved for shared_points
    // int dyn_size_2 = num_src_pts_per_thread * BLOCK_SIZE * sizeof(double);      // memory reserved for min_dis
    // int dyn_size_3 = num_src_pts_per_thread * BLOCK_SIZE * sizeof(int);         // memory reserved for min_ind

    nearest_neighbor_kernel<<<GRID_SIZE, BLOCK_SIZE, (dyn_size_1) >>>(src_device, dst_device, row_src, row_dst, best_neighbor_device, best_dist_device);
    
    

    check_return_status(cudaMemcpy(best_neighbor_host, best_neighbor_device, row_src * sizeof(int), cudaMemcpyDeviceToHost));
    check_return_status(cudaMemcpy(best_dist_host    , best_dist_device    , row_src * sizeof(double), cudaMemcpyDeviceToHost));
    
    for(int i = 0; i < row_src; i++){
        neigh.distances.push_back(best_dist_host[i]);
        neigh.indices.push_back(best_neighbor_host[i]);
    }
    
    //safe_exit();
    free(best_neighbor_host);
    free(best_dist_host);
    cudaFree(src_device);
    cudaFree(dst_device);
    cudaFree(best_neighbor_device);
    cudaFree(best_dist_device);


    // Eigen::Vector3d vec_src;
    // Eigen::Vector3d vec_dst;
    // NEIGHBOR neigh;
    // float min = 100;
    // int index = 0;
    // float dist_temp = 0;

    // for(int ii=0; ii < row_src; ii++){
    //     vec_src = src.block<1,3>(ii,0).transpose();
    //     min = 100;
    //     index = 0;
    //     dist_temp = 0;
    //     for(int jj=0; jj < row_dst; jj++){
    //         vec_dst = dst.block<1,3>(jj,0).transpose();
    //         dist_temp = dist(vec_src,vec_dst);
    //         if (dist_temp < min){
    //             min = dist_temp;
    //             index = jj;
    //         }
    //     }
    //     // cout << min << " " << index << endl;
    //     // neigh.distances[ii] = min;
    //     // neigh.indices[ii] = index;
    //     neigh.distances.push_back(min);
    //     neigh.indices.push_back(index);
    // }

    return neigh;
}