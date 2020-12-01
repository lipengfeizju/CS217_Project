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
NEIGHBOR nearest_neighbor_cuda(const Eigen::MatrixXd &src, const Eigen::MatrixXd &dst);
double   nearest_neighbor_xxxx(const Eigen::MatrixXd &src, const Eigen::MatrixXd &dst);
double ICP_single_step_cuda(const Eigen::MatrixXd &dst,  Eigen::MatrixXd &dst_chorder, const NEIGHBOR &neighbor);
//double ICP_single_step_cuda(const Eigen::MatrixXd &dst, const Eigen::MatrixXd &dst_chorder );
#endif
