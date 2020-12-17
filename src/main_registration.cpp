/*
Registration main function
Code Credit: Pengfei Li
Email: pli081@ucr.edu 
*/
#include <iostream>
#include <vector>
#include <numeric>
#include <sys/time.h>
#include "Eigen/Eigen"
#include "icp.h"
#include "data_io.h"
#include <random>



using namespace std;
using namespace Eigen;


unsigned GetTickCount()
{
        struct timeval tv;
        if(gettimeofday(&tv, NULL) != 0)
                return 0;

        return (tv.tv_sec * 1000) + (tv.tv_usec / 1000);
}


int main(int argc, char *argv[]){

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


    cout << "mean error is " << mean  << endl << endl;
    cout << "iteration: " << iter << endl << endl;
    cout << "icp time: " << total_time <<  " s" <<endl;
    cout << "each iteration takes " << total_time/iter << " s" <<endl;

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




