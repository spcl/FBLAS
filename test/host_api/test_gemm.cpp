/**
  Tests for GEMM routine.
  Tests ideas borrowed from BLAS testing
  GEMM check routine is a modified version of the one included in GSL (Gnu Scientific Library) v2.5
*/
#include <gtest/gtest.h>
#include <string>
#include <exception>
#include <algorithm>
#include <string.h>
#include <cblas.h>
#include "../../include/utils/ocl_utils.hpp"
#include "../../include/fblas_environment.hpp"
#include "test_tier2.hpp"

template <typename T>
void check_result (bool transA, bool transB, unsigned int n, unsigned int m, unsigned int k,
                   T alpha,  T *A, unsigned int lda, T *B, unsigned int ldb,  T beta, T* C, unsigned int ldc, T *fpga_res);

FBLASEnvironment fb;



TEST(TestSGemm,TestSgemm)
{
    cl::CommandQueue queue;
    cl::Context context=fb.get_context();
    cl::Device device=fb.get_device();
    IntelFPGAOCLUtils::createCommandQueue(context,device,queue);
    float *A,*B,*C,*fpga_res;
    const int nalf=3;
    const float alphas[nalf] = {0,1.0,0.7};
    const int nbeta=3;
    const float betas[nbeta] = {0,1.0,0.9};

    posix_memalign ((void **)&A, IntelFPGAOCLUtils::AOCL_ALIGNMENT, N*N*sizeof(float));
    posix_memalign ((void **)&B, IntelFPGAOCLUtils::AOCL_ALIGNMENT, N*N*sizeof(float));
    posix_memalign ((void **)&C, IntelFPGAOCLUtils::AOCL_ALIGNMENT, N*N*sizeof(float));
    posix_memalign ((void **)&fpga_res, IntelFPGAOCLUtils::AOCL_ALIGNMENT, N*N*sizeof(float));

    cl::Buffer input_A(context, CL_MEM_READ_ONLY|CL_CHANNEL_1_INTELFPGA, N * N*sizeof(float));
    cl::Buffer input_B(context, CL_MEM_READ_ONLY|CL_CHANNEL_2_INTELFPGA, N * N*sizeof(float));
    cl::Buffer input_output_C(context, CL_MEM_READ_WRITE|CL_CHANNEL_3_INTELFPGA, N * N*sizeof(float));

    std::string kernels[]={"test_sgemm_notrans_notrans","test_sgemm_notrans_trans",
                           "test_sgemm_trans_notrans","test_sgemm_trans_trans"};

    for(int in=0; in<nd;in++)  //N Dimension loop
    {
        int n=ndim[in];

        for(int im=0; im<nd;im++)  //M Dimension loop
        {
            int m=ndim[im];

            for(int ik=0; ik<nd;ik++)  //K Dimension loop
            {
                int k=ndim[ik];

                for(int icta=0; icta<2;icta++)  //A notrans/trans
                {
                    bool transposedA = (icht[icta]=='T');
                    FblasTranspose transA = (icht[icta]=='T')? FBLAS_TRANSPOSED : FBLAS_NO_TRANSPOSED;

                    //generate A
                    int na=(transposedA)?k:n;
                    int ma=(transposedA)?n:k;
                    int lda=(transposedA)?n:k;

                    generate_matrix<float>(A,na,ma);

                    //copy to device
                    queue.enqueueWriteBuffer(input_A,CL_TRUE,0,na*ma*sizeof(float),A);


                    for(int ictb=0; ictb<2;ictb++)  //B notrans/trans
                    {
                        bool transposedB = (icht[ictb]=='T');
                        FblasTranspose transB = (icht[ictb]=='T')? FBLAS_TRANSPOSED : FBLAS_NO_TRANSPOSED;

                        //generate B
                        int nb=(transposedB)?m:k;
                        int mb=(transposedB)?k:m;
                        int ldb=(transposedB)?k:m;
                        generate_matrix<float>(B,nb,mb);

                        //copy to device
                        queue.enqueueWriteBuffer(input_B,CL_TRUE,0,nb*mb*sizeof(float),B);
                        /*
                        printf("N: %d, M: %d, K: %d",n,m,k);
                        if(transposedA)
                            printf(", A transposed");
                        else
                            printf(", A non transposed");
                        if(transposedB)
                            printf(", B transposed");
                        else
                            printf(", B non transposed");
                        printf("\n");*/


                        for(int ia = 0; ia < nalf ; ia++) //loop over alphas
                        {
                            float alpha=alphas[ia];

                            for(int ib = 0; ib < nbeta ; ib++) //loop over betas
                            {
                                float beta=betas[ib];

                                //generate C
                                generate_matrix<float>(C,n,m);
                                queue.enqueueWriteBuffer(input_output_C,CL_TRUE,0,n*m*sizeof(float),C);
                                fb.sgemm(kernels[icta*2+ictb],transA,transB,n,m,k,alpha,input_A,lda,input_B,ldb,beta,input_output_C,m);

                                //copy back
                                queue.enqueueReadBuffer(input_output_C,CL_TRUE,0,n*m*sizeof(float),fpga_res);

                                check_result<float>(transposedA,transposedB,n,m,k,alpha,A,lda,B,ldb,beta,C,m,fpga_res);
                            }
                        }
                    }
                }
            }
        }
    }
}

TEST(TestDGemm,TestDgemm)
{
    cl::CommandQueue queue;
    cl::Context context=fb.get_context();
    cl::Device device=fb.get_device();
    IntelFPGAOCLUtils::createCommandQueue(context,device,queue);
    double *A,*B,*C,*fpga_res;
    const int nalf=3;
    const double alphas[nalf] = {0,1.0,0.7};
    const int nbeta=3;
    const double betas[nbeta] = {0,1.0,0.9};

    posix_memalign ((void **)&A, IntelFPGAOCLUtils::AOCL_ALIGNMENT, N*N*sizeof(double));
    posix_memalign ((void **)&B, IntelFPGAOCLUtils::AOCL_ALIGNMENT, N*N*sizeof(double));
    posix_memalign ((void **)&C, IntelFPGAOCLUtils::AOCL_ALIGNMENT, N*N*sizeof(double));
    posix_memalign ((void **)&fpga_res, IntelFPGAOCLUtils::AOCL_ALIGNMENT, N*N*sizeof(double));

    cl::Buffer input_A(context, CL_MEM_READ_ONLY|CL_CHANNEL_1_INTELFPGA, N * N*sizeof(double));
    cl::Buffer input_B(context, CL_MEM_READ_ONLY|CL_CHANNEL_2_INTELFPGA, N * N*sizeof(double));
    cl::Buffer input_output_C(context, CL_MEM_READ_WRITE|CL_CHANNEL_3_INTELFPGA, N * N*sizeof(double));

    std::string kernels[]={"test_dgemm_notrans_notrans","test_dgemm_notrans_trans",
                           "test_dgemm_trans_notrans","test_dgemm_trans_trans"};

    for(int in=0; in<nd;in++)  //N Dimension loop
    {
        int n=ndim[in];

        for(int im=0; im<nd;im++)  //M Dimension loop
        {
            int m=ndim[im];

            for(int ik=0; ik<nd;ik++)  //K Dimension loop
            {
                int k=ndim[ik];

                for(int icta=0; icta<2;icta++)  //A notrans/trans
                {
                    bool transposedA = (icht[icta]=='T');
                    FblasTranspose transA = (icht[icta]=='T')? FBLAS_TRANSPOSED : FBLAS_NO_TRANSPOSED;

                    //generate A
                    int na=(transposedA)?k:n;
                    int ma=(transposedA)?n:k;
                    int lda=(transposedA)?n:k;

                    generate_matrix<double>(A,na,ma);

                    //copy to device
                    queue.enqueueWriteBuffer(input_A,CL_TRUE,0,na*ma*sizeof(double),A);


                    for(int ictb=0; ictb<2;ictb++)  //B notrans/trans
                    {
                        bool transposedB = (icht[ictb]=='T');
                        FblasTranspose transB = (icht[ictb]=='T')? FBLAS_TRANSPOSED : FBLAS_NO_TRANSPOSED;

                        //generate B
                        int nb=(transposedB)?m:k;
                        int mb=(transposedB)?k:m;
                        int ldb=(transposedB)?k:m;
                        generate_matrix<double>(B,nb,mb);

                        //copy to device
                        queue.enqueueWriteBuffer(input_B,CL_TRUE,0,nb*mb*sizeof(double),B);


                        for(int ia = 0; ia < nalf ; ia++) //loop over alphas
                        {
                            double alpha=alphas[ia];

                            for(int ib = 0; ib < nbeta ; ib++) //loop over betas
                            {
                                double beta=betas[ib];

                                //generate C
                                generate_matrix<double>(C,n,m);
                                queue.enqueueWriteBuffer(input_output_C,CL_TRUE,0,n*m*sizeof(double),C);
                                fb.dgemm(kernels[icta*2+ictb],transA,transB,n,m,k,alpha,input_A,lda,input_B,ldb,beta,input_output_C,m);

                                //copy back
                                queue.enqueueReadBuffer(input_output_C,CL_TRUE,0,n*m*sizeof(double),fpga_res);

                                check_result<double>(transposedA,transposedB,n,m,k,alpha,A,lda,B,ldb,beta,C,m,fpga_res);
                            }
                        }
                    }
                }
            }
        }
    }
}




int main(int argc, char *argv[])
{
    if(argc<3)
    {
        std::cerr << "Usage: env CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1 " << argv[0] << " <fpga binary file> <json description>" << std::endl;
        return -1;
    }
    std::string program_path=argv[1];
    std::string json_path=argv[2];
    fb =FBLASEnvironment (program_path,json_path);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}




template <typename T>
void check_result (bool transA, bool transB, unsigned int N, unsigned int M, unsigned int K,
                   T alpha,  T *A, unsigned int lda, T *B, unsigned int ldb, T beta, T* C, unsigned int ldc, T *fpga_res)
{

    int NN=N;
    int MM=M;

    if (beta == 0.0) {
        for (int i = 0; i < NN; i++) {
          for (int j = 0; j < MM; j++) {
            C[ldc * i + j] = 0.0;
          }
        }
      } else if (beta != 1.0) {
        for (int i = 0; i < NN; i++) {
          for (int j = 0; j < MM; j++) {
            C[ldc * i + j] *= beta;
          }
        }
      }
    if(!transA && !transB)
    {
        for (int k = 0; k < K; k++) {
            for (int i = 0; i < NN; i++) {
                const T temp = alpha * A[lda * i + k];
                if (temp != 0.0) {
                    for (int j = 0; j < MM; j++) {
                        C[ldc * i + j] += temp * B[ldb * k + j];
                    }
                }
            }
        }
    }

    if(!transA && transB)
    {
        for (int i = 0; i < NN; i++) {
            for (int j = 0; j < MM; j++) {
                T temp = 0.0;
                for (int k = 0; k < K; k++) {
                    temp += A[lda * i + k] * B[ldb * j + k];
                }
                C[ldc * i + j] += alpha * temp;
            }
        }
    }

    if(transA && !transB)
    {
        for (int k = 0; k < K; k++) {
            for (int i = 0; i < NN; i++) {
                const T temp = alpha * A[lda * k + i];
                if (temp != 0.0) {
                    for (int j = 0; j < MM; j++) {
                        C[ldc * i + j] += temp * B[ldb * k + j];
                    }
                }
            }
        }
    }
    if(transA && transB)
    {
        for (int i = 0; i < NN; i++) {
            for (int j = 0; j < MM; j++) {
                T temp = 0.0;
                for (int k = 0; k < K; k++) {
                    temp += A[lda * k + i] * B[ldb * j + k];
                }
                C[ldc * i + j] += alpha * temp;
            }
        }
    }


CHECK:
    //print_float_matrix(fpga_res,NN,MM);
    //print_float_matrix(C,NN,MM);
//    Measure error by considering nrm_inf
    T nrminf_diff=0, nrminf_orig=0;
    T error;

    for(int i=0;i < NN; i++)
    {
        T nrminf=0, nrminf_o=0;
        for(int j=0; j<MM;j++)
        {
            nrminf+=abs(fpga_res[i*ldc+j]-C[i*ldc+j]);
            nrminf_o+=abs(C[i*ldc+j]);
        }
        if(nrminf>nrminf_diff)
            nrminf_diff=nrminf;
        if(nrminf_o>nrminf_orig)
            nrminf_orig=nrminf_o;
    }
    if(nrminf_diff==0 && nrminf_orig ==0)
        error=0;
    else
        error=nrminf_diff/nrminf_orig;

    if (std::is_same<T, float>::value)
        ASSERT_LE(error,flteps);
    else
        ASSERT_LE(error,dbleps);
}
