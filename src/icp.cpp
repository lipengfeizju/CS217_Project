#include <iostream>
#include <numeric>
#include "icp.h"
#include "Eigen/Eigen"

using namespace std;
using namespace Eigen;



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

    JacobiSVD<Eigen::MatrixXd> svd(H, ComputeFullU | ComputeFullV);
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

/*
typedef struct{
    Eigen::Matrix4d trans;
    std::vector<float> distances;
    int iter;
}  ICP_OUT;
*/

ICP_OUT icp(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B, int max_iterations, float tolerance){
    int row = A.rows();
    Eigen::MatrixXd src = Eigen::MatrixXd::Ones(3+1,row);
    Eigen::MatrixXd src3d = Eigen::MatrixXd::Ones(3,row);
    //Eigen::MatrixXd dst = Eigen::MatrixXd::Ones(3+1,row);

    NEIGHBOR neighbor;
    Eigen::Matrix4d T;
    Eigen::MatrixXd dst_chorder = Eigen::MatrixXd::Ones(3,row);
    ICP_OUT result;
    int iter = 0;


    src(Eigen::seqN(0,3), Eigen::all) = A.transpose();
    src3d(Eigen::seqN(0,3), Eigen::all) = A.transpose();

    Eigen::MatrixXd dst = B.transpose();


    double prev_error = 0;
    double mean_error = 0;
    for (int i=0; i<max_iterations; i++){

        neighbor = nearest_neighbor(src3d.transpose(), dst.transpose());

        // Calculate mean error and compare with previous error
        mean_error = std::accumulate(neighbor.distances.begin(),neighbor.distances.end(),0.0)/neighbor.distances.size();
        if (abs(prev_error - mean_error) < tolerance){
            break;
        }

        dst_chorder(Eigen::seqN(0,3), all) = dst(Eigen::seqN(0,3), neighbor.indices);

        T = transform_SVD(src3d.transpose(),dst_chorder.transpose());

        src = T*src;
        // Copy first 3 rows to src3d
        src3d(all, all) = src(seqN(0,3), all);

        prev_error = mean_error;
        iter = i+2;
    }

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
        min = 100;
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
        // cout << min << " " << index << endl;
        // neigh.distances[ii] = min;
        // neigh.indices[ii] = index;
        neigh.distances.push_back(min);
        neigh.indices.push_back(index);
    }

    return neigh;
}


float dist(const Eigen::Vector3d &pta, const Eigen::Vector3d &ptb){
    return sqrt((pta[0]-ptb[0])*(pta[0]-ptb[0]) + (pta[1]-ptb[1])*(pta[1]-ptb[1]) + (pta[2]-ptb[2])*(pta[2]-ptb[2]));
}
