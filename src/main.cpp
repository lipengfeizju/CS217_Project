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


    std::random_device rd;
    std::mt19937 gen(rd());

    std::string file_name = "/home/anny/cuda-workspace/icp_project/icp/data/samples/airplane_0001.txt";
    MatrixXd pcl_data = load_pcl(file_name);
    int num_point = pcl_data.rows();

    MatrixXd A = pcl_data;
    Vector3d t;
    Matrix3d R;
    Matrix4d T;    // Transformation result from ICP
    Vector3d t1;
    Matrix3d R1;
    ICP_OUT icp_result;
    std::vector<float> dist;
    int iter;
    float mean;

    float total_time = 0;
    unsigned start, end;
    float interval;

    /***************       Construct Random Matrix B      ******************/
    MatrixXd B = A;
    t = Vector3d::Random()*translation;

    for( int jj =0; jj< num_point; jj++){
        B.block<1,3>(jj,0) = B.block<1,3>(jj,0) + t.transpose();
    }

    R = rotation_matrix(Vector3d::Random() ,my_random()*rotation);
    B = (R * B.transpose()).transpose();

    B += MatrixXd::Random(num_point,3) * noise_sigma;

    // shuffle
    my_random_shuffle(B);
    
    save_pcl("/home/anny/cuda-workspace/icp_project/icp/data/samples/airplane_0001_rotated.txt",B);

    /******** Calculate ICP ***********/
    start = GetTickCount();
    icp_result = icp(B, A, 50,  1e-6);
    end = GetTickCount();
    interval = float((end - start))/1000;
    total_time += interval;

    T = icp_result.trans;
    dist = icp_result.distances;
    iter = icp_result.iter;
    mean = std::accumulate(dist.begin(),dist.end(),0.0)/dist.size();


    t1 = T.block<3,1>(0,3);
    R1 = T.block<3,3>(0,0);


    cout << "mean error is " << mean - 6*noise_sigma << endl << endl;
    cout << "icp trans error" << endl << -t1 - t << endl << endl;
    cout << "icp R error " << endl << R1.inverse() - R << endl << endl;
    cout << "total iteration is " << iter << endl;
    
    cout << "icp time: " << total_time << endl;

    /**********  Reconstruct the point cloud    *************/

    MatrixXd D = B;
    D = (R1 * D.transpose()).transpose();
    D.rowwise() += t1.transpose();


    std::cout << "Writing recovered point cloud data to file" << std::endl;
    save_pcl("/home/anny/cuda-workspace/icp_project/icp/data/samples/airplane_0001_recovered.txt",D);

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





