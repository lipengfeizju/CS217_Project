#include "Eigen/Eigen"
#include <vector>
#include "icp.h"
#ifndef ICP_CUDA_H
#define ICP_CUDA_H

// CUDA speedup method
NEIGHBOR nearest_neighbor_cuda(const Eigen::MatrixXf &src, const Eigen::MatrixXf &dst);

double apply_optimal_transform_cuda(const Eigen::MatrixXf &dst,  const Eigen::MatrixXf &src, Eigen::MatrixXf &src_transformed, const NEIGHBOR &neighbor);
double single_step_ICP(const Eigen::MatrixXf &dst,  const Eigen::MatrixXf &src, const NEIGHBOR &neighbor, Eigen::MatrixXf &src_transformed, NEIGHBOR &neighbor_out);
int icp_cuda(const Eigen::MatrixXf &dst,  const Eigen::MatrixXf &src, int max_iterations, float tolerance, Eigen::MatrixXf &src_transformed, NEIGHBOR &neighbor_out);
//double ICP_single_step_cuda(const Eigen::MatrixXd &dst, const Eigen::MatrixXd &dst_chorder );
#endif
