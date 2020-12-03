#include "Eigen/Eigen"
#include <vector>

#ifndef ICP_H
#define ICP_H

//#define N_tests 100    // # of test iterations
#define noise_sigma 1e-4    // standard deviation error to be added
#define translation 1     // max translation of the test set
#define rotation 1        // max rotation (radians) of the test set
#define INF 1e40

typedef struct{
    Eigen::Matrix4d trans;
    std::vector<float> distances;
    int iter;
}  ICP_OUT;

typedef struct{
    std::vector<float> distances;
    std::vector<int> indices;
} NEIGHBOR;

Eigen::Matrix4d transform_SVD(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B);

ICP_OUT icp(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B, int max_iterations=20, float tolerance = 0.001);

// throughout method
NEIGHBOR nearest_neighbor(const Eigen::MatrixXd &src, const Eigen::MatrixXd &dst);
float dist(const Eigen::Vector3d &pta, const Eigen::Vector3d &ptb);

// CUDA speedup method
NEIGHBOR nearest_neighbor_cuda(const Eigen::MatrixXf &src, const Eigen::MatrixXf &dst);

double apply_optimal_transform_cuda(const Eigen::MatrixXf &dst,  const Eigen::MatrixXf &src, Eigen::MatrixXf &src_transformed, const NEIGHBOR &neighbor);
double single_step_ICP(const Eigen::MatrixXf &dst,  const Eigen::MatrixXf &src, const NEIGHBOR &neighbor, Eigen::MatrixXf &src_transformed, NEIGHBOR &neighbor_out);
//double ICP_single_step_cuda(const Eigen::MatrixXd &dst, const Eigen::MatrixXd &dst_chorder );
#endif
