/*
Utility functions for data-io
Code Credit: Pengfei Li
Email: pli081@ucr.edu
*/
#include "Eigen/Eigen"

#define MAXBUFSIZE  ((int) 1e6)

using namespace Eigen;

MatrixXd load_pcl(std::string file_name, int col = 3);
void save_pcl(std::string file_name, MatrixXd& pcl_data);
void save_tranformation(std::string file_name, Matrix4d& transformation);