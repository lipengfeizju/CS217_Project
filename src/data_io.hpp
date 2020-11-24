#include "Eigen/Eigen"

#define MAXBUFSIZE  ((int) 1e6)

using namespace Eigen;

MatrixXd load_pcl(char *file_name, int col = 3);
void save_pcl(char *file_name, MatrixXd& pcl_data);