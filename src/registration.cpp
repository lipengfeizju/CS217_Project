#include <iostream>
#include <vector>
#include <numeric>
#include <sys/time.h>
#include "Eigen/Eigen"
#include "icp.h"
#include "data_io.hpp"
#include <random>



using namespace std;
using namespace Eigen;



float my_random(void);
Matrix3d rotation_matrix(Vector3d axis, float theta);
void test_best_fit(void);
void test_icp(void);
void my_random_shuffle(MatrixXd &matrix);


unsigned GetTickCount()
{
        struct timeval tv;
        if(gettimeofday(&tv, NULL) != 0)
                return 0;

        return (tv.tv_sec * 1000) + (tv.tv_usec / 1000);
}


int main(int argc, char *argv[]){


    // std::random_device rd;
    // std::mt19937 gen(rd());

    // std::string base_dir = "/home/anny/cuda-workspace/icp_project/icp/data/samples/";
    // std::cout << base_dir + argv[1] <<std::endl;
    // MatrixXd A = load_pcl(base_dir + "airplane_0001.txt");    
    // MatrixXd B = load_pcl(base_dir + "airplane_0001_rotated.txt");
    MatrixXd A, B;
    std::string base_dir = "";
    if (argc == 4){
        base_dir += argv[1];
        A = load_pcl(base_dir + argv[2]);
        B = load_pcl(base_dir + argv[3]);
    }
    else{
        printf("usage: ./registration  path_to_base_dir original_file_path translated_file_path");
        exit(0);
    }
    

    float total_time = 0;
    unsigned start, end;
    float interval;


    ICP_OUT icp_result;
    

    /******** Calculate ICP ***********/
    start = GetTickCount();
    icp_result = icp(B, A, 50,  1e-6);
    end = GetTickCount();
    interval = float((end - start))/1000;
    total_time += interval;


    Matrix4d T = icp_result.trans;
    std::vector<float> dist = icp_result.distances;
    int iter = icp_result.iter;
    float mean = std::accumulate(dist.begin(),dist.end(),0.0)/dist.size();


    Vector3d t1 = T.block<3,1>(0,3);
    Matrix3d R1 = T.block<3,3>(0,0);


    cout << "mean error is " << mean - 6*noise_sigma << endl << endl;
    cout << "icp time: " << total_time << endl;

    /**********  Reconstruct the point cloud    *************/

    MatrixXd D = B;
    D = (R1 * D.transpose()).transpose();
    D.rowwise() += t1.transpose();


    std::cout << "Writing recovered point cloud data to file" << std::endl;
    save_pcl(base_dir + "recovered.txt",D);

    std::cout << "Writing estimated transformation to file" << std::endl;
    save_tranformation(base_dir + "transformation.txt", T);
    return 0;
}



///////////////////////////
//  help function

// 0-1 float variables
float my_random(void){
    float tmp = rand()%100;
    return tmp/100;
}

void my_random_shuffle(MatrixXd &matrix){
    int row = matrix.rows();
    vector<Vector3d> temp;
    for(int jj=0; jj < row; jj++){
        temp.push_back(matrix.block<1,3>(jj,0));
    }
    random_shuffle(temp.begin(),temp.end());
    for(int jj=0; jj < row; jj++){
        matrix.block<1,3>(jj,0) = temp[jj].transpose();
        // cout << temp[jj].transpose() << endl;
        // cout << "row  " << row << endl;
    }
}


Matrix3d rotation_matrix(Vector3d axis, float theta){
    axis = axis / sqrt(axis.transpose()*axis);
    float a = cos(theta/2);
    Vector3d temp = -axis*sin(theta/2);
    float b,c,d;
    b = temp(0);
    c = temp(1);
    d = temp(2);
    Matrix3d R;
    R << a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c),
        2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b),
        2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c;

    return R;
}





