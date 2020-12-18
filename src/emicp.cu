// EM_ICP Wendy Liu

#include <iostream>
#include <numeric>
#include <cmath>
#include "icp.h"    // shared with icp
#include "Eigen/Eigen"
#include <assert.h>
#include <iomanip>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include "support.cu"
#include "emicp.h"
#include <Eigen/Dense>
#define BLOCK_SIZE 32
#define GRID_SIZE 128



__global__ static void updateA(int rowsA, int colsA, int pitchA,
	const float* d_Xx, const float* d_Xy, const float* d_Xz, 
	const float* d_Yx, const float* d_Yy, const float* d_Yz,
	const float* d_R, const float* d_t,
	float* d_A, float sigma_p2){

  int r =  blockIdx.x * blockDim.x + threadIdx.x;
  int c =  blockIdx.y * blockDim.y + threadIdx.y;

  // Shared memory
  __shared__ float XxShare[BLOCK_SIZE];
  __shared__ float XyShare[BLOCK_SIZE];
  __shared__ float XzShare[BLOCK_SIZE];
  __shared__ float YxShare[BLOCK_SIZE];
  __shared__ float YyShare[BLOCK_SIZE];
  __shared__ float YzShare[BLOCK_SIZE];
  __shared__ float RShare[9];
  __shared__ float tShare[3];

  if(threadIdx.x == 0 && threadIdx.y == 0)
  {
    for (int i = 0; i < 9; i++) RShare[i] = d_R[i];
    for (int i = 0; i < 3; i++) tShare[i] = d_t[i];
  }
  
  if(r < rowsA && c < colsA){ // check for only inside the matrix A

    if(threadIdx.x == 0){
      XxShare[threadIdx.y] = d_Xx[c];
      XyShare[threadIdx.y] = d_Xy[c];
      XzShare[threadIdx.y] = d_Xz[c];
    }
    if(threadIdx.y == 0){
      YxShare[threadIdx.x] = d_Yx[r];
      YyShare[threadIdx.x] = d_Yy[r];
      YzShare[threadIdx.x] = d_Yz[r];
    }

    __syncthreads();

  #define Xx XxShare[threadIdx.y]
  #define Xy XyShare[threadIdx.y]
  #define Xz XzShare[threadIdx.y]
  #define Yx YxShare[threadIdx.x]
  #define Yy YyShare[threadIdx.x]
  #define Yz YzShare[threadIdx.x]
  #define R(i) RShare[i]
  #define t(i) tShare[i]

  // #define Euclid(a,b,c) ((a)*(a)+(b)*(b)+(c)*(c))
  //     float tmp =
  //       Euclid(Xx - (R(0)*Yx + R(1)*Yy + R(2)*Yz + t(0)),
  //              Xy - (R(3)*Yx + R(4)*Yy + R(5)*Yz + t(1)),
  //              Xz - (R(6)*Yx + R(7)*Yy + R(8)*Yz + t(2)) );
      
  //     tmp = expf(-tmp/sigma_p^2)


     float tmpX = Xx - (R(0)*Yx + R(1)*Yy + R(2)*Yz + t(0));
     float tmpY = Xy - (R(3)*Yx + R(4)*Yy + R(5)*Yz + t(1));
     float tmpZ = Xz - (R(6)*Yx + R(7)*Yy + R(8)*Yz + t(2));

    __syncthreads();

     tmpX *= tmpX;
     tmpY *= tmpY;
     tmpZ *= tmpZ;

     tmpX += tmpY;
     tmpX += tmpZ;

     tmpX /= sigma_p2;
     tmpX = expf(-tmpX);

    //float *A = (float*)((char*)d_A + c * pitchMinBytes) + r;

    d_A[c * pitchA + r] = tmpX;
  }
}

__global__ static void normalizeRowsOfA(int rowsA, int colsA, int pitchA, float *d_A, const float *d_C){
  
  int r =  blockIdx.x * blockDim.x + threadIdx.x;
  int c =  blockIdx.y * blockDim.y + threadIdx.y;

  // Shared memory
  __shared__ float d_CShare[BLOCK_SIZE];


  if(r < rowsA && c < colsA){ // check for only inside the matrix A

    if(threadIdx.y == 0)
      d_CShare[threadIdx.x] = d_C[r];

    __syncthreads();

    if(d_CShare[threadIdx.x] > 10e-7f)
      // each element in A is normalized C, then squre-rooted
      d_A[c * pitchA + r] = sqrtf( d_A[c * pitchA + r] / d_CShare[threadIdx.x] );
    else
      d_A[c * pitchA + r] = 1.0f/colsA; // ad_hoc code to avoid 0 division

    __syncthreads();

  }
}

__global__ static void elementwiseDivision(int Xsize,
		    float* d_Xx, float* d_Xy, float* d_Xz,
		    const float* d_lambda){

  int x =  blockIdx.x * blockDim.x + threadIdx.x;

  if(x < Xsize){
    float l_lambda = d_lambda[x];
    d_Xx[x] /= l_lambda;
    d_Xy[x] /= l_lambda;
    d_Xz[x] /= l_lambda;
  }
}

__global__ static void elementwiseMultiplication(int Xsize, float* d_Xx, float* d_Xy, float* d_Xz, const float* d_lambda){

  int x =  blockIdx.x * blockDim.x + threadIdx.x;

  if(x < Xsize){
    float l_lambda = d_lambda[x];
    d_Xx[x] *= l_lambda;
    d_Xy[x] *= l_lambda;
    d_Xz[x] *= l_lambda;
  }
}

__global__ static void centeringXandY(int rowsA,
         const float* d_Xc, const float* d_Yc,
	       const float* d_Xx, const float* d_Xy, const float* d_Xz,
	       const float* d_Yx, const float* d_Yy, const float* d_Yz,
	       float* d_XxCenterd, float* d_XyCenterd, float* d_XzCenterd,
	       float* d_YxCenterd, float* d_YyCenterd, float* d_YzCenterd
	       ){

  // do for both X and Y at the same time
  
  int r =  blockIdx.x * blockDim.x + threadIdx.x;

  // Shared memory
  __shared__ float Xc[3];
  __shared__ float Yc[3];

  if(threadIdx.x < 6) // assume blocksize >= 6
    if(threadIdx.x < 3) 
      Xc[threadIdx.x] = d_Xc[threadIdx.x];
    else
      Yc[threadIdx.x - 3] = d_Yc[threadIdx.x - 3];


  if(r < rowsA){

    __syncthreads();

    d_XxCenterd[r] = d_Xx[r] - Xc[0];
    d_XyCenterd[r] = d_Xy[r] - Xc[1];
    d_XzCenterd[r] = d_Xz[r] - Xc[2];

    d_YxCenterd[r] = d_Yx[r] - Yc[0];
    d_YyCenterd[r] = d_Yy[r] - Yc[1];
    d_YzCenterd[r] = d_Yz[r] - Yc[2];

    __syncthreads();

  }
}
extern "C" {
  int dsyev_(char *jobz, char *uplo, 
       int *n, double *a, int *lda, 
       double *w, double *work, int *lwork, 
       int *info);
  }
void eigenvectorOfN(double *N, float* q){
  
  static float q_pre[4]; // previous result

  int dimN = 4;
  double w[4]; // eigenvalues
  double *work = new double; // workspace
  int info;
  int lwork = -1;

  // dsyev_((char*)"V", (char*)"U",
	//  &dimN, N, &dimN,
	//  w, work, &lwork, &info);
  // if(info != 0){
  //   fprintf(stderr, "info = %d\n", info);
  //   exit(1);
  // }
  // lwork = (int)work[0];
  // delete work;

  // work = new double [lwork];

  // dsyev_((char*)"V", (char*)"U",
	//  &dimN, N, &dimN,
	//  w, work, &lwork, &info);

  // delete [] work;


  if(info != 0){
    fprintf(stderr, "computing eigenvector FAIL! info = %d\n", info);
    //exit(1);

    // if fail, put back the previous result
    for(int i=0; i<4; i++){
      q[i] = q_pre[i];
    }
  }else{
    // last column of N is the eigenvector of the largest eigenvalue 
    // and N is stored column-major
    for(int i=0; i<4; i++){
      q[i] = N[4*3 + i];
      q_pre[i] = q[i];
    }
  }
}

void findRTfromS(const float* h_Xc, const float* h_Yc, const float* h_S, float* h_R, float* h_t){

  #define h_Sxx h_S[0]
  #define h_Sxy h_S[1]
  #define h_Sxz h_S[2]
  #define h_Syx h_S[3]
  #define h_Syy h_S[4]
  #define h_Syz h_S[5]
  #define h_Szx h_S[6]
  #define h_Szy h_S[7]
  #define h_Szz h_S[8]

  #define h_Xcx h_Xc[0]
  #define h_Xcy h_Xc[1]
  #define h_Xcz h_Xc[2]
  #define h_Ycx h_Yc[0]
  #define h_Ycy h_Yc[1]
  #define h_Ycz h_Yc[2]


  double N[4*4]; for(int n=0;n<16;n++) N[n] = 0.0;
  float q[4];    for(int a=0;a<4;a++)  q[a] = 0.0f;

  N[ 0] = h_Sxx + h_Syy + h_Szz;
  N[ 1] = h_Syz - h_Szy;
  N[ 2] = h_Szx - h_Sxz;
  N[ 3] = h_Sxy - h_Syx;
  N[ 4] = h_Syz - h_Szy;
  N[ 5] = h_Sxx - h_Syy - h_Szz;
  N[ 6] = h_Sxy + h_Syx;
  N[ 7] = h_Szx + h_Sxz;
  N[ 8] = h_Szx - h_Sxz;
  N[ 9] = h_Sxy + h_Syx;
  N[10] = h_Syy - h_Sxx - h_Szz;
  N[11] = h_Syz + h_Szy;
  N[12] = h_Sxy - h_Syx;
  N[13] = h_Szx + h_Sxz;
  N[14] = h_Syz + h_Szy;
  N[15] = h_Szz - h_Sxx - h_Syy;

  // compute the eigenvector corresponding the largest eigenvalue
  // eigenvectorOfN(N, q);
  // Eigen::ComplexEigenSolver<Eigen::Matrix<std::complex<double>, 2,2> > s(A);
  // Eigen::MatrixXd newN = Eigen::Map<Eigen::MatrixXd>(N, 4, 4);
  // Eigen::Matrix<std::complex<double>, 4,4> s(newN);
  // s.eigenvalues();
  for(int i =0; i<16; i++) N[i] = 1+2*i;
  Eigen::MatrixXd A_test = Eigen::Map<Eigen::Matrix<double, 4, 4> >(N);
  Eigen::EigenSolver <Eigen::MatrixXd> eigensolver(A_test);
  if (eigensolver.info() != Eigen::Success) abort();
   std::cout << "The eigenvalues of A are:\n" << eigensolver.eigenvalues() << std::endl;
   std::cout << "Here's a matrix whose columns are eigenvectors of A \n"
        << "corresponding to these eigenvalues:\n"
        << eigensolver.eigenvectors() << std::endl;
  // std::cout << aaaa<< std::endl;
  // for(int n=0;n<16;n++) std::cout << N[n] << ", "<<std::endl;
  printf("So far so good\n");
  exit(0);

  float q0 = q[0], qx = q[1], qy = q[2], qz = q[3];

  // quaternion to rotation matrix
  h_R[0] = q0*q0 + qx*qx - qy*qy - qz*qz;
  h_R[1] = 2 * (qx*qy - q0*qz);
  h_R[2] = 2 * (qx*qz + q0*qy);
  h_R[3] = 2 * (qy*qx + q0*qz);
  h_R[4] = q0*q0 - qx*qx + qy*qy - qz*qz;
  h_R[5] = 2 * (qy*qz - q0*qx);
  h_R[6] = 2 * (qz*qx - q0*qy);
  h_R[7] = 2 * (qz*qy + q0*qx);
  h_R[8] = q0*q0 - qx*qx - qy*qy + qz*qz;

  // translation vector
  h_t[0] = h_Xcx - (h_R[0]*h_Ycx + h_R[1]*h_Ycy + h_R[2]*h_Ycz);
  h_t[1] = h_Xcy - (h_R[3]*h_Ycx + h_R[4]*h_Ycy + h_R[5]*h_Ycz);
  h_t[2] = h_Xcz - (h_R[6]*h_Ycx + h_R[7]*h_Ycy + h_R[8]*h_Ycz);
}

// ----------------------------------- Main --------------------------------

void emicp(const Eigen::MatrixXf cloud_target, const Eigen::MatrixXf cloud_source,
    float* h_R, float* h_t)
{
  float sigma_p2 = 0.01;      // initial value for the main loop. sigma_p2 <- sigma_p2 * sigma_factor  at the end of each iteration while sigma_p2 > sigam_inf. default: 0.01
  float sigma_inf = 0.00001;     // minimum value of sigma_p2. default: 0.00001
  float sigma_factor = 0.9;  // facfor for reducing sigma_p2. default: 0.9
  float d_02 = 0.01;          // values for outlier (see EM-ICP paper). default: 0.01

  int Xsize = cloud_source.rows();
  int Ysize = cloud_source.cols();
  const float *h_X = cloud_source.data();
  const float *h_Y = cloud_target.data();

// Reusable snippets 
// Copied from Tamaki's GitHub repo
#define memCUDA(var,num) float* d_ ## var; cudaMalloc((void**) &(d_ ## var), sizeof(float)*num);

#define memHostToCUDA(var,num) \
float* d_ ## var; cudaMalloc((void**) &(d_ ## var), sizeof(float)*num); \
cudaMemcpy(d_ ## var, h_ ## var, sizeof(float)*num, cudaMemcpyHostToDevice);

// Memory allocation
memHostToCUDA(X, Xsize*3);
float* d_Xx = &d_X[Xsize*0];
float* d_Xy = &d_X[Xsize*1];
float* d_Xz = &d_X[Xsize*2];

memHostToCUDA(Y, Ysize*3);
float* d_Yx = &d_Y[Ysize*0];
float* d_Yy = &d_Y[Ysize*1];
float* d_Yz = &d_Y[Ysize*2];

memCUDA(Xprime, Ysize*3);
float *d_XprimeX = &d_Xprime[Ysize*0];
float *d_XprimeY = &d_Xprime[Ysize*1];
float *d_XprimeZ = &d_Xprime[Ysize*2];

float *d_XprimeCenterd = d_Xprime;
float *d_XprimeCenterdX = &d_XprimeCenterd[Ysize*0];
float *d_XprimeCenterdY = &d_XprimeCenterd[Ysize*1];
float *d_XprimeCenterdZ = &d_XprimeCenterd[Ysize*2];

memCUDA(YCenterd, Ysize*3);
float *d_YCenterdX = &d_YCenterd[Ysize*0];
float *d_YCenterdY = &d_YCenterd[Ysize*1];
float *d_YCenterdZ = &d_YCenterd[Ysize*2];

// center of X, Y
float h_Xc[3], h_Yc[3];
memCUDA(Xc, 3);
memCUDA(Yc, 3);

// R, t
memHostToCUDA(R, 3*3);
memHostToCUDA(t, 3);
cudaMemcpy(d_R, h_R, sizeof(float)*3*3, cudaMemcpyHostToDevice);
cudaMemcpy(d_t, h_t, sizeof(float)*3,   cudaMemcpyHostToDevice);

// S for finding R, t
float h_S[9];
memCUDA(S, 9);


// NOTE on matrix A (from Tamaki)
// number of rows:     Ysize, or rowsA
// number of columns : Xsize, or colsA
// 
//                    [0th in X] [1st]  ... [(Xsize-1)] 
// [0th point in Y] [ A(0,0)     A(0,1) ... A(0,Xsize-1)      ] 
// [1st           ] [ A(1,0)     A(1,1) ...                   ]
// ...              [ ...                                     ]
// [(Ysize-1)     ] [ A(Ysize-1, 0)     ... A(Ysize-1,Xsize-1)]
//
// 
// CAUTION on matrix A
// A is allcoated as a column-maijor format for the use of cublas.
// This means that you must acces an element at row r and column c as:
// A(r,c) = A[c * pitchA + r]

int rowsA = Ysize;
int colsA = Xsize;

// pitchA: leading dimension of A, which is ideally equal to rowsA,
//          but actually larger than that.
int pitchA = (rowsA / 4 + 1) * 4;

memCUDA(A, pitchA*colsA);

// a vector with all elements of 1.0f
float* h_one = new float [max(Xsize,Ysize)];
for(int t = 0; t < max(Xsize,Ysize); t++) h_one[t] = 1.0f;
memHostToCUDA(one, max(Xsize,Ysize));

memCUDA(sumOfMRow, rowsA);
memCUDA(C, rowsA); // sum of a row in A
memCUDA(lambda, rowsA); // weight of a row in A

// for 2D block
dim3 dimBlockForA(BLOCK_SIZE, BLOCK_SIZE); // a block is (BLOCK_SIZE*BLOCK_SIZE) threads
dim3 dimGridForA( (pitchA + dimBlockForA.x - 1) / dimBlockForA.x,
         (colsA  + dimBlockForA.y - 1) / dimBlockForA.y);

// for 1D block
int threadsPerBlockForYsize = 512; // a block is 512 threads
int blocksPerGridForYsize
 = (Ysize + threadsPerBlockForYsize - 1 ) / threadsPerBlockForYsize;

cublasHandle_t handle;
cublasCreate(&handle);

// EM-ICP main loop
// int Titer = 1;
while(sigma_p2 > sigma_inf) {
// UpdateA
   updateA <<< dimGridForA, dimBlockForA >>> (rowsA, colsA, pitchA,
    d_Xx, d_Xy, d_Xz, d_Yx, d_Yy, d_Yz, d_R, d_t, d_A, sigma_p2);
// Normalization of A
   //
   // A * one vector = vector with elements of row-wise sum
   //     d_A      *    d_one    =>  d_C
   //(rowsA*colsA) *  (colsA*1)  =  (rowsA*1)
   float alpha = 1.0;
   float beta = 0.0;
   cublasSgemv(handle, CUBLAS_OP_N, rowsA, colsA, &alpha, d_A, pitchA, d_one, 1, &beta, d_C, 1);

   alpha = expf(-d_02/sigma_p2);
   cublasSaxpy(handle, rowsA, &alpha, d_one, 1, d_C, 1);
   
   normalizeRowsOfA <<< dimGridForA, dimBlockForA >>> (rowsA, colsA, pitchA, d_A, d_C);

   // update R,T
   ///////////////////////////////////////////////////////////////////////////////////// 
   // compute lambda
   
   // A * one vector = vector with elements of row-wise sum
   //     d_A      *    d_one    =>  d_lambda
   //(rowsA*colsA) *  (colsA*1)  =  (rowsA*1)
   // cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, K, M, &alpha, A, M, A, M, &beta, B, N);
    // A(MxN) K = N  A'(N,M)
   cublasSgemv(handle, CUBLAS_OP_N, rowsA, colsA, &alpha, d_A, pitchA, d_one, 1, &beta, d_lambda, 1);

   // cublasStatus_t  cublasSasum(cublasHandle_t handle, int n, const float *x, int incx, float  *result) 
  //  cublasDasum(handle, num_data_pts, best_dist_device, 1, &prev_error)
   float result = 0; // place-holder
   float sumLambda = cublasSasum (handle, rowsA, d_lambda, 1, &result);
   ///////////////////////////////////////////////////////////////////////////////////// 
   // compute X'
   // m      number of rows of matrix op(A) and rows of matrix C
   // n      number of columns of matrix op(B) and number of columns of C
   // k      number of columns of matrix op(A) and number of rows of op(B) 

   // A * X => X'
   //     d_A      *    d_X    =>  d_Xprime
   //(rowsA*colsA) *  (colsA*3)  =  (rowsA*3)
   //   m  * k           k * n        m * n   
   alpha = 1;
   beta = 0;
   cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, rowsA, 3, colsA, &alpha, d_A, pitchA, d_X, colsA, &beta, d_Xprime, rowsA);
   // X' ./ lambda => X'
   elementwiseDivision <<< blocksPerGridForYsize, threadsPerBlockForYsize>>> (rowsA, d_XprimeX, d_XprimeY, d_XprimeZ, d_lambda);
   ///////////////////////////////////////////////////////////////////////////////////// 

   //
   // centering X' and Y
   //
   ///////////////////////////////////////////////////////////////////////////////////// 
   // find weighted center of X' and Y
   // d_Xprime^T *    d_lambda     =>   h_Xc
   //  (3 * rowsA)   (rowsA * 1)  =  (3 * 1)
   cublasSgemv(handle, CUBLAS_OP_T, rowsA, 3, &alpha, d_Xprime, rowsA, d_lambda, 1, &beta, d_Xc, 1);

   // d_Y^T *    d_lambda     =>   h_Yc
   //  (3 * rowsA)   (rowsA * 1)  =  (3 * 1)
   cublasSgemv(handle, CUBLAS_OP_T, rowsA, 3, &alpha, d_Y, rowsA, d_lambda, 1, &beta, d_Yc, 1);

   // void cublasSscal (int n, float alpha, float *x, int incx)
   // it replaces x[ix + i * incx] with alpha * x[ix + i * incx]
   alpha = 1/sumLambda;
   cublasSscal (handle, 3, &alpha, d_Xc, 1);
   cublasSscal (handle, 3, &alpha, d_Yc, 1);

   cudaMemcpy(h_Xc, d_Xc, sizeof(float)*3, cudaMemcpyDeviceToHost);
   cudaMemcpy(h_Yc, d_Yc, sizeof(float)*3, cudaMemcpyDeviceToHost);
   ///////////////////////////////////////////////////////////////////////////////////// 

   // centering X and Y
   // d_Xprime .- d_Xc => d_XprimeCenterd
   // d_Y      .- d_Yc => d_YCenterd
   centeringXandY <<< blocksPerGridForYsize, threadsPerBlockForYsize>>> (rowsA, 
    d_Xc, d_Yc, d_XprimeX, d_XprimeY, d_XprimeZ, d_Yx, d_Yy, d_Yz, 
    d_XprimeCenterdX, d_XprimeCenterdY, d_XprimeCenterdZ, d_YCenterdX, d_YCenterdY, d_YCenterdZ);

   // XprimeCented .* d_lambda => XprimeCented
   elementwiseMultiplication <<< blocksPerGridForYsize, threadsPerBlockForYsize>>>
    (rowsA, d_XprimeCenterdX, d_XprimeCenterdY, d_XprimeCenterdZ, d_lambda);
   ///////////////////////////////////////////////////////////////////////////////////// 
   // compute S
   //  d_XprimeCented^T *   d_YCenterd     =>  d_S
   //    (3*rowsA)  *  (rowsA*3)  =  (3*3)
   //   m  * k           k * n        m * n
   cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 3, 3, rowsA, &alpha, d_XprimeCenterd, rowsA, d_YCenterd, rowsA, &beta, d_S, 3);

   cudaMemcpy(h_S, d_S, sizeof(float)*9, cudaMemcpyDeviceToHost);
   ///////////////////////////////////////////////////////////////////////////////////// 
   // find RT from S
   findRTfromS(h_Xc, h_Yc, h_S, h_R, h_t);
  //  STOP_TIMER(timerAfterSVD);
   ///////////////////////////////////////////////////////////////////////////////////// 
   // copy R,t to device
   cudaMemcpy(d_R, h_R, sizeof(float)*3*3, cudaMemcpyHostToDevice);
   cudaMemcpy(d_t, h_t, sizeof(float)*3,   cudaMemcpyHostToDevice);
   ///////////////////////////////////////////////////////////////////////////////////// 
 sigma_p2 *= sigma_factor;
}
cudaDeviceSynchronize();
cublasDestroy(handle);

cudaFree(d_X);
cudaFree(d_Y);
cudaFree(d_Xprime);
cudaFree(d_YCenterd);
cudaFree(d_Xc);
cudaFree(d_Yc);

cudaFree(d_R);
cudaFree(d_t);
cudaFree(d_A);

cudaFree(d_S);
cudaFree(d_one);
cudaFree(d_sumOfMRow);
cudaFree(d_C);
cudaFree(d_lambda);

delete [] h_one;
}
