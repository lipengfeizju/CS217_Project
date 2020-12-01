#include <iostream>
#include <numeric>
#include "icp.h"
#include "Eigen/Eigen"

#define USE_GPU false
using namespace std;

void verify(NEIGHBOR neigbor1, NEIGHBOR neigbor2){
    if(!(neigbor1.distances.size() == neigbor2.distances.size() && 
         neigbor1.indices.size() == neigbor2.indices.size() &&
         neigbor1.distances.size() == neigbor1.indices.size()
        )){
        std::cout << "Neighbor size not match" <<std::endl;
        exit(-1);
    }
    int num_pts = neigbor1.distances.size();
    for(int i = 0; i < num_pts; i++){
        if(neigbor1.distances[i] - neigbor2.distances[i] > 1e-20 ||
            neigbor1.indices[i] != neigbor2.indices[i]){
            std::cout << "Neighbor result not match: " << i << std::endl;
            std::cout << "neighbor 1 " << neigbor1.distances[i] << ", " << neigbor1.indices[i] << std::endl;
            std::cout << "neighbor 2 " << neigbor2.distances[i] << ", " << neigbor2.indices[i] << std::endl;
            exit(-1);
        }
    }

}

void verify(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B){
    if(A.rows() != B.rows() || A.cols() != B.cols()){
        std::cout << "Matrix shape not match" << std::endl;
        std::cout << "A :   " << A.rows() <<", "<< A.cols() << std::endl;
        std::cout << "B :   " << B.rows() <<", "<< B.cols() << std::endl;
        exit(-1);
    }
    int row = A.rows(), col = B.cols();
    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            if(A(i,j) != B(i,j)){
                std::cout << i << ", " << j << ": " << A(i,j) << ", " << B(i,j) << std::endl;
                exit(-1);
            }
        }
    }
}


ICP_OUT icp(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B, int max_iterations, float tolerance){
    int row = A.rows();
    Eigen::MatrixXd src = Eigen::MatrixXd::Ones(3+1,row);
    Eigen::MatrixXd src3d = Eigen::MatrixXd::Ones(3,row);
    //Eigen::MatrixXd dst = Eigen::MatrixXd::Ones(3+1,row);

    NEIGHBOR neighbor;
    //NEIGHBOR neighbor_cpu;
    Eigen::Matrix4d T;
    Eigen::MatrixXd dst_chorder = Eigen::MatrixXd::Ones(3,row);
    ICP_OUT result;
    int iter = 0;


    src(Eigen::seqN(0,3), Eigen::all) = A.transpose();
    src3d(Eigen::seqN(0,3), Eigen::all) = A.transpose();

    Eigen::MatrixXd dst = B.transpose();


    double prev_error = 0;
    double mean_error = 0;
    if(USE_GPU)
        neighbor = nearest_neighbor_cuda(src3d.transpose(), dst.transpose());
    else
        neighbor = nearest_neighbor(src3d.transpose(), dst.transpose());


    prev_error = std::accumulate(neighbor.distances.begin(),neighbor.distances.end(),0.0)/neighbor.distances.size();

    double temp = 0;
    Eigen::MatrixXd dst_chorder_test = Eigen::MatrixXd::Ones(3, row);
    Eigen::MatrixXd dst_chorder_temp = Eigen::MatrixXd::Ones(row, 3);


    for (int i=0; i<max_iterations; i++){

        dst_chorder(Eigen::seqN(0,3), Eigen::all) = dst(Eigen::seqN(0,3), neighbor.indices);
                
        T = transform_SVD(src3d.transpose(),dst_chorder.transpose());

        src = T*src;
        // Copy first 3 rows to src3d
        src3d(Eigen::all, Eigen::all) = src(Eigen::seqN(0,3), Eigen::all);


        if(USE_GPU)
            neighbor = nearest_neighbor_cuda(src3d.transpose(), dst.transpose());
        else
            neighbor = nearest_neighbor(src3d.transpose(), dst.transpose());
        
        mean_error = std::accumulate(neighbor.distances.begin(),neighbor.distances.end(),0.0)/neighbor.distances.size();


        std::cout << mean_error << std::endl;
        if (abs(prev_error - mean_error) < tolerance){
            break;
        }
        // Calculate mean error and compare with previous error
        
        prev_error = mean_error;
        iter = i+2;
    }
    std::cout << tolerance << std::endl;

    T = transform_SVD(A,src3d.transpose());
    result.trans = T;
    result.distances = neighbor.distances;
    result.iter = iter;

    return result;
}

/*
typedef struct{
    std::vector<float> distances;
    std::vector<int> indices;
} NEIGHBOR;
*/

NEIGHBOR nearest_neighbor(const Eigen::MatrixXd &src, const Eigen::MatrixXd &dst){
    int row_src = src.rows();
    int row_dst = dst.rows();
    Eigen::Vector3d vec_src;
    Eigen::Vector3d vec_dst;
    NEIGHBOR neigh;
    float min = 100;
    int index = 0;
    float dist_temp = 0;

    for(int ii=0; ii < row_src; ii++){
        vec_src = src.block<1,3>(ii,0).transpose();
        min = INF;
        index = 0;
        dist_temp = 0;
        for(int jj=0; jj < row_dst; jj++){
            vec_dst = dst.block<1,3>(jj,0).transpose();
            dist_temp = dist(vec_src,vec_dst);
            if (dist_temp < min){
                min = dist_temp;
                index = jj;
            }
        }

        neigh.distances.push_back(min);
        neigh.indices.push_back(index);
    }

    return neigh;
}

Eigen::Matrix4d transform_SVD(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B){
    /*
     * min ||B- RA||
    Notice:
    1/ JacobiSVD return U,S,V, S as a vector, "use U*S*Vt" to get original Matrix;
    2/ matrix type 'MatrixXd' or 'MatrixXf' matters.
    */
    Eigen::Matrix4d T = Eigen::MatrixXd::Identity(4,4);
    Eigen::Vector3d centroid_A(0,0,0);
    Eigen::Vector3d centroid_B(0,0,0);
    Eigen::MatrixXd A_zm = A; // Zero mean version of A
    Eigen::MatrixXd B_zm = B; // Zero mean version of B
    int row = A.rows();


    centroid_A = A.colwise().sum() / row;
    centroid_B = B.colwise().sum() / row;


    A_zm.rowwise() -= centroid_A.transpose();
    B_zm.rowwise() -= centroid_B.transpose();


    Eigen::MatrixXd H = A_zm.transpose()*B_zm;
    Eigen::MatrixXd U;
    Eigen::VectorXd S;
    Eigen::MatrixXd V;
    Eigen::MatrixXd Vt;
    Eigen::Matrix3d R;
    Eigen::Vector3d t;

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    U = svd.matrixU();
    S = svd.singularValues();
    V = svd.matrixV();
    Vt = V.transpose();

    R = Vt.transpose()*U.transpose();


    if (R.determinant() < 0 ){
        Vt.block<1,3>(2,0) *= -1;
        R = Vt.transpose()*U.transpose();
    }

    t = centroid_B - R*centroid_A;

    T.block<3,3>(0,0) = R;
    T.block<3,1>(0,3) = t;
    return T;

}

float dist(const Eigen::Vector3d &pta, const Eigen::Vector3d &ptb){
    return sqrt((pta[0]-ptb[0])*(pta[0]-ptb[0]) + (pta[1]-ptb[1])*(pta[1]-ptb[1]) + (pta[2]-ptb[2])*(pta[2]-ptb[2]));
}
