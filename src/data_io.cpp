#include <iostream>
#include <fstream>
#include "Eigen/Eigen"
#include "data_io.h"
#include <assert.h>
using namespace Eigen;


MatrixXd load_pcl(std::string file_name, int col ){

    int counter = 0;

    std::ifstream point_cloud_file(file_name);
    if (!point_cloud_file.good()){
        std::cout << "File not found:  " << file_name << std::endl;
        exit(0);
    }
    

    std::string line;
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

    std::cout << "MODEL OPENED FROM: " << file_name << std::endl;
    return pcl_matrix;
}

void save_pcl(std::string file_name, MatrixXd& pcl_data){
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
    else std::cout << "Unable to open file  " << file_name << std::endl;

}

void save_tranformation(std::string file_name, Matrix4d& transformation){
    std::ofstream out_file (file_name);
    int cols = transformation.cols();
    int rows = transformation.rows();

    // out_file << "Transformation:  " << std::endl;
    if (out_file.is_open())
    {
        for(int i = 0; i < rows; i++){
            for(int j = 0; j < cols-1; j++){
                out_file << transformation(i,j) << ", ";
            }
            out_file << transformation(i,cols-1);
            if(i < rows -1)
                out_file << std::endl;
        }
        out_file.close();
    }
    else std::cout << "Unable to open file  " << file_name << std::endl;

}