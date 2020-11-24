#include <iostream>
#include <fstream>
#include "Eigen/Eigen"
#include "data_io.hpp"

using namespace Eigen;


MatrixXd load_pcl(char *file_name, int col ){

    int counter = 0;

    std::ifstream point_cloud_file(file_name);
    std::string line;
    printf("OPENING MODEL \n");
    double buffer[MAXBUFSIZE];

    std::string temp;

    while (std::getline(point_cloud_file, line)) {

        std::stringstream ss(line);
        for( int j = 0; j < col; j++){
            std::getline(ss, temp, ',');
            buffer[col*counter + j] = std::stod(temp);
        }
        counter++;
    }

    point_cloud_file.close();
    MatrixXd pcl_matrix(counter,col);
    for(int i =0; i < counter; i++){
        for(int j = 0; j < col; j++)
            pcl_matrix(i,j) = buffer[col*i + j];
    }

    return pcl_matrix;
}

void save_pcl(char *file_name, MatrixXd& pcl_data){
    std::ofstream pcl_file (file_name);
    int cols = pcl_data.cols();
    int rows = pcl_data.rows();

    if (pcl_file.is_open())
    {
        for(int i = 0; i < rows; i++){
            for(int j = 0; j < cols-1; j++){
                pcl_file << pcl_data(i,j) << ", ";
            }
            pcl_file << pcl_data(i,cols-1);
            if(i < rows -1)
                pcl_file << std::endl;
        }
        pcl_file.close();
    }
    else std::cout << "Unable to open file";

}