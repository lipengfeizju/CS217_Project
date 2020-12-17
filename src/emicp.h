#include "Eigen/Eigen"

#ifndef EMICP_H
#define EMICP_H
void emicp(const Eigen::MatrixXf cloud_target, const Eigen::MatrixXf cloud_source,
    float* h_R, float* h_t);
void eigenvectorOfN(double* N, float* q);
void findRTfromS(const float* h_Xc, const float* h_Yc, const float* h_S, float* h_R, float* h_t);
#endif
