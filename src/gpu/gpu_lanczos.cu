// simple gpu lanczos with cublas
#include <stdio.h>
#include <stdlib.h>
#include "assert.h"
#include <vector>

#include "mkl.h"

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

using namespace std;

//sub_a[i]=a[i*sub_size : (i+1)*sub_size]
__global__ void gpu_generate_sub_matrix(double * a, double ** sub_a, int sub_size){
    int t_id=blockIdx.x * blockDim.x + threadIdx.x;
    sub_a[t_id]=&a[sub_size*t_id];
    return;
}

//a=sqrt(a)
__global__ void gpu_sqrt_kernel(double * a){
    int t_id=blockIdx.x * blockDim.x + threadIdx.x;
    a[t_id]=sqrt(a[t_id]);
    return;
}

//a=1/sqrt(a)
__global__ void gpu_rsqrt_kernel(double * a){
    int t_id=blockIdx.x * blockDim.x + threadIdx.x;
    a[t_id]=rsqrt(a[t_id]);
    return;
}

//a=a-b
__global__ void gpu_minus_kernel(double * a,double * b){
    int t_id=blockIdx.x * blockDim.x + threadIdx.x;
    a[t_id]=a[t_id]-b[t_id];
    return;
}

//a=0
__global__ void gpu_zero_kernel(double * a){
    int t_id=blockIdx.x * blockDim.x + threadIdx.x;
    a[t_id]=0;
    return;
}

//ia=1/a
__global__ void gpu_inv_kernel(double * ia, double * a){
    int t_id=blockIdx.x * blockDim.x + threadIdx.x;
    ia[t_id]=1/a[t_id];
    return;
}

//ans=A*B
void gpu_dot(cublasHandle_t & handle, int n,\
                double * A_d, int inca,\
                double * B_d, int incb,\
                double * ans_d)
{
    cublasStatus_t status;
    status = cublasDdot(handle, n,A_d,inca,B_d,incb,ans_d);
    return;
}

//ans=sqrt(A*A)
void gpu_norm(cublasHandle_t & handle, int n,\
                double * A_d, int inca,\
                double * ans_d)
{
    gpu_dot(handle, n,A_d,inca,A_d,inca,ans_d);
    gpu_sqrt_kernel<<<1,1>>>(ans_d);
    return;
}

//ans=1/sqrt(A*A)
void gpu_inorm(cublasHandle_t & handle, int n,\
                double * A_d, int inca,\
                double * ans_d)
{
    gpu_dot(handle, n,A_d,inca,A_d,inca,ans_d);
    gpu_rsqrt_kernel<<<1,1>>>(ans_d);
    return;
}

//X = a*X
void gpu_scal(cublasHandle_t & handle, int n,\
    double * a_d, double * X_d, int incx)
{
    cublasStatus_t status;
    status = cublasDscal(handle, n, a_d, X_d, incx);
    return;
}

//X = X/a
void gpu_iscal(cublasHandle_t & handle, int n,\
    double * a_d, double * X_d, int incx)
{
    cublasStatus_t status;
    double * ia_d;
    cudaMalloc(reinterpret_cast<void**>(&ia_d ), sizeof(ia_d[0] ));
    gpu_inv_kernel<<<1,1>>>(ia_d,a_d);
    status = cublasDscal(handle, n, ia_d, X_d, incx);
    cudaFree(ia_d);
    return;
}

//Y = a*X+Y
void gpu_axpy(cublasHandle_t & handle, int n,\
    double * a_d, double * X_d, int incx,\
    double * Y_d, int incy)
{
    cublasStatus_t status;
    status = cublasDaxpy(handle, n, a_d, X_d, incx, Y_d, incy);
    return;
}

//Y = -a*X+Y
void gpu_maxpy(cublasHandle_t & handle, int n,\
    double * a_d, double * X_d, int incx,\
    double * Y_d, int incy)
{
    cublasStatus_t status;
    double * ma_d;
    cudaMalloc(reinterpret_cast<void**>(&ma_d ), sizeof(ma_d[0] ));
    gpu_zero_kernel<<<1,1>>>(ma_d);
    gpu_minus_kernel<<<1,1>>>(ma_d,a_d);
    status = cublasDaxpy(handle, n, ma_d, X_d, incx, Y_d, incy);
    cudaFree(ma_d);
    return;
}

//Y = X/a+Y
void gpu_iaxpy(cublasHandle_t & handle, int n,\
    double * a_d, double * X_d, int incx,\
    double * Y_d, int incy)
{
    cublasStatus_t status;
    double * ia_d;
    cudaMalloc(reinterpret_cast<void**>(&ia_d ), sizeof(ia_d[0] ));
    gpu_inv_kernel<<<1,1>>>(ia_d,a_d);
    status = cublasDaxpy(handle, n, ia_d, X_d, incx, Y_d, incy);
    cudaFree(ia_d);
    return;
}

void gpu_Heffv_cublas(cublasHandle_t & handle,\
                        double * ENVL_MPO_d, int * ENVL_MPO_dim,\
                        double * ENVR_d, int * ENVR_dim,\
                        double * V0_d, int * V0_dim,\
                        double * V1_d)
{
    /*****************************************************
    *       Heff*v = ENVL_MPO * V0 *ENVR = V1
    *
    *         |0   |1                      |0   |1    
    *        ENVL_MPO--2 *           =>   ENVL_MPO--2
    *         |    |4          |1          |    |   
    *         |--3          0--V0--2       |----V0--3  
    * 
    *
    *         |0   |1          0|           |1
    *        ENVL_MPO--2 *  1--ENVR  =>  0--V1--2
    *         |    |            |
    *         |----V0--3      2-|
    *****************************************************/
    cublasStatus_t status;
    cublasOperation_t transa,transb;
    transa=CUBLAS_OP_N;
    transb=CUBLAS_OP_N;
    double one=1.0;
    double zero=0.0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    double * ENVL_MPO_MPS_d;
    size_t envl_mpo_mps_size=ENVL_MPO_dim[0]*ENVL_MPO_dim[1]*ENVL_MPO_dim[2]*V0_dim[2];
    cudaMalloc(reinterpret_cast<void**>(&ENVL_MPO_MPS_d ), envl_mpo_mps_size  * sizeof(ENVL_MPO_MPS_d[0] ));

    //ENVL_MPO * V0
    cudaEventRecord(start, 0);
    status = cublasDgemm(handle, transa, transb,
        ENVL_MPO_dim[0]*ENVL_MPO_dim[1]*ENVL_MPO_dim[2], V0_dim[2], V0_dim[0]*V0_dim[1], &one, ENVL_MPO_d, ENVL_MPO_dim[0]*ENVL_MPO_dim[1]*ENVL_MPO_dim[2], V0_d, V0_dim[0]*V0_dim[1], &zero, ENVL_MPO_MPS_d, ENVL_MPO_dim[0]*ENVL_MPO_dim[1]*ENVL_MPO_dim[2]);

    //ENVL_MPO_MPS * ENVR
    transb=CUBLAS_OP_T;
    status = cublasDgemm(handle, transa, transb,
        ENVL_MPO_dim[0]*ENVL_MPO_dim[1], ENVR_dim[0], ENVL_MPO_dim[2]*V0_dim[2], &one, ENVL_MPO_MPS_d, ENVL_MPO_dim[0]*ENVL_MPO_dim[1], ENVR_d, ENVR_dim[0], &zero, V1_d, ENVL_MPO_dim[0]*ENVL_MPO_dim[1]);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop); 
    float time;
    cudaEventElapsedTime(&time, start, stop);

    cudaFree(ENVL_MPO_MPS_d);
}

void gpu_ENVL_MPO_cublas(cublasHandle_t & handle,\
                        double * MPO_d, int * MPO_dim,\
                        double * ENVL_d, int * ENVL_dim,\
                        double * ENVL_MPO_d)
{
    /*****************************************************
    *       generate ENVL_MPO
    *           |-- 0        1|           |0   |1
    *       ENVL--- 1     0--MPO--2  =>  ENVL_MPO--2
    *           |-- 2        3|           |3   |4
    *****************************************************/
    cublasStatus_t status;
    double one=1.0;
    double zero=0.0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cublasOperation_t transa,transb;
    transa=CUBLAS_OP_N;
    transb=CUBLAS_OP_N;

    double ** sub_ENVL_d;
    double ** sub_MPO_d;
    double ** sub_ENVL_MPO_d;

    cudaMalloc(reinterpret_cast<void**>(&sub_ENVL_d ), ENVL_dim[2]  * sizeof(sub_ENVL_d[0]));
    cudaMalloc(reinterpret_cast<void**>(&sub_MPO_d ), MPO_dim[3]  * sizeof(sub_MPO_d[0]));
    cudaMalloc(reinterpret_cast<void**>(&sub_ENVL_MPO_d ), ENVL_dim[2]*MPO_dim[3]  * sizeof(sub_ENVL_MPO_d[0]));
    gpu_generate_sub_matrix<<< ENVL_dim[2]*MPO_dim[3]/64,64>>>(ENVL_MPO_d,sub_ENVL_MPO_d,ENVL_dim[0]*MPO_dim[1]*MPO_dim[2]);
    gpu_generate_sub_matrix<<<ENVL_dim[2]/64,64>>>(ENVL_d,sub_ENVL_d,ENVL_dim[0]*ENVL_dim[1]);
    gpu_generate_sub_matrix<<<MPO_dim[3]/64,64>>>(MPO_d,sub_MPO_d,MPO_dim[0]*MPO_dim[1]*MPO_dim[2]);

    cudaEventRecord(start, 0);
    for(int d6=0;d6<MPO_dim[3];d6++){
        for(int d3=0;d3<ENVL_dim[2];d3++){
            status = cublasDgemm(handle, transa, transb,
                ENVL_dim[0], MPO_dim[1]*MPO_dim[2], ENVL_dim[1], &one, sub_ENVL_d[d3], ENVL_dim[0], sub_MPO_d[d6], MPO_dim[0], &zero, sub_ENVL_MPO_d[d6*ENVL_dim[2]+d3], ENVL_dim[0]);
        }
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop); 
    float time;
    cudaEventElapsedTime(&time, start, stop);

    cudaFree(sub_MPO_d);
    cudaFree(sub_ENVL_d);
    cudaFree(sub_ENVL_MPO_d);

}

void gpu_simple_lanczos(double * MPO, int * MPO_dim,\
                        double * ENVL, int * ENVL_dim,\
                        double * ENVR, int * ENVR_dim,\
                        double * MPS, int * MPS_dim,\
                        double * eigval, double * eigvac, int *info,\
                        int k_max=10, double btol=1.0e-8, int verbosity=0)
{
    cublasHandle_t handle;
    cublasStatus_t status;
	status = cublasCreate(&handle);
    *info = 0;
    
    double * MPO_d;
    double * ENVL_d;
    double * ENVL_MPO_d;

    size_t mpo_size=MPO_dim[0]*MPO_dim[1]*MPO_dim[2]*MPO_dim[3];
    size_t envl_size=ENVL_dim[0]*ENVL_dim[1]*ENVL_dim[2];
    size_t envl_mpo_size=ENVL_dim[0]*MPO_dim[1]*MPO_dim[2]*ENVL_dim[2]*MPO_dim[3];

    int ENVL_MPO_dim[5];
    ENVL_MPO_dim[0]=ENVL_dim[0];
    ENVL_MPO_dim[1]=MPO_dim[1];
    ENVL_MPO_dim[2]=MPO_dim[2];
    ENVL_MPO_dim[3]=ENVL_dim[2];
    ENVL_MPO_dim[4]=MPO_dim[3];

    cudaMalloc(reinterpret_cast<void**>(&MPO_d ), mpo_size  * sizeof(MPO_d[0] ));
    cudaMalloc(reinterpret_cast<void**>(&ENVL_d), envl_size * sizeof(ENVL_d[0]));
    cudaMalloc(reinterpret_cast<void**>(&ENVL_MPO_d ), envl_mpo_size  * sizeof(ENVL_MPO_d[0] ));
    
    status = cublasSetVector(mpo_size  , sizeof(MPO[0] ), MPO , 1, MPO_d , 1);
    status = cublasSetVector(envl_size , sizeof(ENVL[0]), ENVL, 1, ENVL_d, 1);

    gpu_ENVL_MPO_cublas(handle,\
                        MPO_d, MPO_dim,\
                        ENVL_d, ENVL_dim,\
                        ENVL_MPO_d);

    cudaFree(MPO_d);
    cudaFree(ENVL_d);

    /*****************************************************
    *       simple lanczos first step
    *****************************************************/

    double * ENVR_d;
    //double * MPS_d;
    double * tmpV0_d;
    double * tmpV1_d;
    double * tmpV2_d;
    vector<double *> V;

    size_t envr_size=ENVR_dim[0]*ENVR_dim[1]*ENVR_dim[2];
    size_t mps_size=MPS_dim[0]*MPS_dim[1]*MPS_dim[2];

    cudaMalloc(reinterpret_cast<void**>(&ENVR_d), envr_size * sizeof(ENVR_d[0]));
    cudaMalloc(reinterpret_cast<void**>(&tmpV0_d ), mps_size  * sizeof(tmpV0_d[0] ));
    cudaMalloc(reinterpret_cast<void**>(&tmpV1_d ), mps_size  * sizeof(tmpV1_d[0] ));

    status = cublasSetVector(envr_size , sizeof(ENVR[0]), ENVR, 1, ENVR_d, 1);
    status = cublasSetVector(mps_size  , sizeof(MPS[0] ), MPS , 1, tmpV0_d , 1);

    double * ivm_d;
    double * a_d;
    double * b_d;
    double a,b;
    cudaMalloc(reinterpret_cast<void**>(&ivm_d), sizeof(ivm_d[0]));
    cudaMalloc(reinterpret_cast<void**>(&a_d), sizeof(a_d[0]));
    cudaMalloc(reinterpret_cast<void**>(&b_d), sizeof(b_d[0]));

    //a, b, qiminus1, qi, vm = lanczos_make_first_step(A, v)
    // MPS = MPS / norm(MPS)
    gpu_inorm(handle,mps_size,tmpV0_d,1,ivm_d);
    gpu_scal(handle,mps_size,ivm_d,tmpV0_d,1);
    
    //qi = A(qiminus1)
    gpu_Heffv_cublas(handle,\
                        ENVL_MPO_d, ENVL_MPO_dim,\
                        ENVR_d, ENVR_dim,\
                        tmpV0_d, MPS_dim,\
                        tmpV1_d);
    //a = dot(qiminus1, qi)
    gpu_dot(handle,mps_size,tmpV0_d,1,tmpV1_d,1,a_d);
    //qi -= qiminus1 * a
    gpu_maxpy(handle,mps_size,a_d,tmpV0_d,1,tmpV1_d,1);
    //b = norm(qi)
    gpu_norm(handle,mps_size,tmpV1_d,1,b_d);

    status = cublasGetVector(1, sizeof(double), a_d, 1, &a, 1);
    status = cublasGetVector(1, sizeof(double), b_d, 1, &b, 1);
    if(b!=0.0){
        gpu_iscal(handle,mps_size,b_d,tmpV1_d,1);
    }
    if(b < btol){
        if(verbosity > 3) printf("lanczos converged after the first iteration");
        eigval[0]=a;
        status = cublasGetVector(mps_size, sizeof(eigvac[0]), tmpV0_d, 1, eigvac, 1);
        return;
    }
    V.push_back(tmpV0_d);

    double * T;
    double * lambda;
    T=(double *)calloc(k_max*k_max,sizeof(double));
    lambda=(double *)calloc(k_max,sizeof(double));
    T[0]=a;
    int k=1;
    bool converged = false;

    while(k<k_max){
        T[(k-1)*k_max+k]=b;
        T[k*k_max+k-1]=b;

        //a, b, qiminus1, qi = lanczos_make_step(A, qiminus1, qi, b)
        cudaMalloc(reinterpret_cast<void**>(&tmpV2_d ), mps_size  * sizeof(tmpV2_d[0] ));
        //qiplus1 = A(qi)
        gpu_Heffv_cublas(handle,\
                        ENVL_MPO_d, ENVL_MPO_dim,\
                        ENVR_d, ENVR_dim,\
                        tmpV1_d, MPS_dim,\
                        tmpV2_d);
        //a = dot(qi, qiplus1)
        gpu_dot(handle,mps_size,tmpV1_d,1,tmpV2_d,1,a_d);
        //qiplus1 -= (qi*a + qiminus1 * bold)
        gpu_maxpy(handle,mps_size,a_d,tmpV1_d,1,tmpV2_d,1);
        gpu_axpy(handle,mps_size,b_d,tmpV0_d,1,tmpV2_d,1);
        //b = norm(qiplus1)
        gpu_norm(handle,mps_size,tmpV2_d,1,b_d);

        status = cublasGetVector(1, sizeof(double), a_d, 1, &a, 1);
        status = cublasGetVector(1, sizeof(double), b_d, 1, &b, 1);
        if(b!=0.0){
            gpu_iscal(handle,mps_size,b_d,tmpV2_d,1);
        }
        tmpV0_d=tmpV1_d;
        tmpV1_d=tmpV2_d;

        V.push_back(tmpV0_d);
        T[k*k_max+k]=a;
        k++;
        if(b < btol){
            if(verbosity > 3) printf("lanczos converges after %d iterations",k);
            converged=true;
            break;
        }
    }
    if(!converged){
        *info=-1;
        if(verbosity > 3) printf("lanczos fail to converge after %d iterations",k_max);
    }

    //eigen(Symmetric(T[1:k, 1:k]))
    LAPACKE_dsyev( LAPACK_COL_MAJOR, 'V', 'U', k, T, k_max, lambda );
    eigval[0]=lambda[0];

    //lanczos_linear_transformation_single
    double * eigvac_d;
    cudaMalloc(reinterpret_cast<void**>(&eigvac_d ), mps_size  * sizeof(eigvac_d[0] ));
    gpu_zero_kernel<<<mps_size/64,64>>>(eigvac_d);
    for(int i=0;i<k;i++){
        status = cublasSetVector(1 , sizeof(a_d[0]), &T[i*k_max], 1, a_d, 1);
        gpu_axpy(handle,mps_size,a_d,V[i],1,eigvac_d,1);
    }
    status = cublasGetVector(mps_size, sizeof(eigvac[0]), eigvac_d, 1, eigvac, 1);
    
    cudaFree(ENVL_MPO_d);
    cudaFree(ENVR_d);
    cudaFree(tmpV0_d);
    cudaFree(tmpV1_d);
    for(int i=0;i<k;i++){
        cudaFree(V[i]);
    }
    cudaFree(tmpV1_d);
    cudaFree(eigvac_d);
    cudaFree(a_d);
    cudaFree(b_d);
    status = cublasDestroy(handle);
    
    return;
}

void gpu_ac_prime(  double * x, int * x_dim,\
                    double * mpojhleft, int * mpojhleft_dim,\
                    double * hright, int * hright_dim,\
                    double * ans)
{
    cublasHandle_t handle;
    cublasStatus_t status;
	status = cublasCreate(&handle);

    double * x_d;
    double * mpojhleft_d;
    double * hright_d;
    double * ans_d;

    size_t x_size=x_dim[0]*x_dim[1]*x_dim[2];
    size_t mpojhleft_size=mpojhleft_dim[0]*mpojhleft_dim[1]*mpojhleft_dim[2]*mpojhleft_dim[3]*mpojhleft_dim[4];
    size_t hright_size=hright_dim[0]*hright_dim[1]*hright_dim[2];

    cudaMalloc(reinterpret_cast<void**>(&x_d), x_size * sizeof(x_d[0]));
    cudaMalloc(reinterpret_cast<void**>(&ans_d), x_size * sizeof(ans_d[0]));
    cudaMalloc(reinterpret_cast<void**>(&mpojhleft_d ), mpojhleft_size  * sizeof(mpojhleft_d[0] ));
    cudaMalloc(reinterpret_cast<void**>(&hright_d ), hright_size  * sizeof(hright_d[0] ));

    status = cublasSetVector(mpojhleft_size , sizeof(mpojhleft_d[0]), mpojhleft, 1, mpojhleft_d, 1);
    status = cublasSetVector(hright_size  , sizeof(hright_d[0] ), hright , 1, hright_d , 1);
    status = cublasSetVector(x_size  , sizeof(x_d[0] ), x , 1, x_d , 1);

    gpu_Heffv_cublas(handle,\
                        mpojhleft_d, mpojhleft_dim,\
                        hright_d, hright_dim,\
                        x_d, x_dim,\
                        ans_d);
    
    status = cublasGetVector(x_size, sizeof(ans_d[0]), ans_d, 1, ans, 1);

    cudaFree(x_d);
    cudaFree(mpojhleft_d);
    cudaFree(hright_d);
    cudaFree(ans_d);
    status = cublasDestroy(handle);
}
    

