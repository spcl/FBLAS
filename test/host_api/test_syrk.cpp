/**
  Tests for SYRK routine.
  Tests ideas borrowed from BLAS testing
  SYRK check routine is a modified version of the one included in GSL (Gnu Scientific Library) v2.5
*/
#include <gtest/gtest.h>
#include <string>
#include <exception>
#include <algorithm>
#include <string.h>
#include "../../include/utils/ocl_utils.hpp"
#include "../../include/fblas_environment.hpp"
#include "test_tier2.hpp"

template <typename T>
void check_result (bool transA, bool lowerC, unsigned int n, unsigned int k,
                   T alpha,  T *A, unsigned int lda,  T beta, T* C, unsigned int ldc, T *fpga_res);

FBLASEnvironment fb;


TEST(TestSSyrk,TestSsyrk)
{
    cl::CommandQueue queue;
    cl::Context context=fb.get_context();
    cl::Device device=fb.get_device();
    IntelFPGAOCLUtils::createCommandQueue(context,device,queue);
    float *A,*C,*fpga_res;
    const int nalf=3;
    const float alphas[nalf] = {0,1.0,0.7};
    const int nbeta=3;
    const float betas[nbeta] = {0,1.0,0.9};

    posix_memalign ((void **)&A, IntelFPGAOCLUtils::AOCL_ALIGNMENT, N*N*sizeof(float));
    posix_memalign ((void **)&C, IntelFPGAOCLUtils::AOCL_ALIGNMENT, N*N*sizeof(float));
    posix_memalign ((void **)&fpga_res, IntelFPGAOCLUtils::AOCL_ALIGNMENT, N*N*sizeof(float));

    cl::Buffer input_A(context, CL_MEM_READ_ONLY|CL_CHANNEL_1_INTELFPGA, N * N*sizeof(float));
    cl::Buffer input_output_C(context, CL_MEM_READ_WRITE|CL_CHANNEL_3_INTELFPGA, N * N*sizeof(float));

    std::string kernels[]={"test_ssyrk_notrans_lower","test_ssyrk_notrans_upper","test_ssyrk_trans_lower","test_ssyrk_trans_upper"};

    for(int in=0; in<nd;in++)  //N Dimension loop
    {
        int n=ndim[in];

        for(int ik=0; ik<nd;ik++)  //K Dimension loop
        {
            int k=ndim[ik];

            for(int icta=0; icta<2;icta++)  //A notrans/trans
            {
                bool transposedA = (icht[icta]=='T');
                FblasTranspose transA = (icht[icta]=='T')? FBLAS_TRANSPOSED : FBLAS_NO_TRANSPOSED;

                //generate A
                int na=(transposedA)?k:n;
                int ka=(transposedA)?n:k;
                int lda=(transposedA)?n:k;

                generate_matrix<float>(A,na,ka);

                //copy to device
                queue.enqueueWriteBuffer(input_A,CL_TRUE,0,na*ka*sizeof(float),A);


                for(int icu=0; icu<2;icu++)  //C lower/upper
                {

                    for(int ia = 0; ia < nalf ; ia++) //loop over alphas
                    {
                        float alpha=alphas[ia];

                        for(int ib = 0; ib < nbeta ; ib++) //loop over betas
                        {
                            float beta=betas[ib];
                            bool lowerC = (ichu[icu]=='L');
                            FblasUpLo uploC=(ichu[icu]=='L')?FBLAS_LOWER:FBLAS_UPPER;

                            //generate C
                            generate_matrix<float>(C,n,n);

                            //generate C
                            queue.enqueueWriteBuffer(input_output_C,CL_TRUE,0,n*n*sizeof(float),C);

                            fb.ssyrk(kernels[icta*2+icu],uploC,transA,n,k,alpha,input_A,lda,beta,input_output_C,n);

                            //copy back
                            queue.enqueueReadBuffer(input_output_C,CL_TRUE,0,n*n*sizeof(float),fpga_res);

                            check_result<float>(transposedA,lowerC,n,k,alpha,A,lda,beta,C,n,fpga_res);
                        }
                    }
                }
            }
        }
    }
}


TEST(TestDSyrk,TestDsyrk)
{
    cl::CommandQueue queue;
    cl::Context context=fb.get_context();
    cl::Device device=fb.get_device();
    IntelFPGAOCLUtils::createCommandQueue(context,device,queue);
    double *A,*C,*fpga_res;
    const int nalf=3;
    const double alphas[nalf] = {0,1.0,0.7};
    const int nbeta=3;
    const double betas[nbeta] = {0,1.0,0.9};

    posix_memalign ((void **)&A, IntelFPGAOCLUtils::AOCL_ALIGNMENT, N*N*sizeof(double));
    posix_memalign ((void **)&C, IntelFPGAOCLUtils::AOCL_ALIGNMENT, N*N*sizeof(double));
    posix_memalign ((void **)&fpga_res, IntelFPGAOCLUtils::AOCL_ALIGNMENT, N*N*sizeof(double));

    cl::Buffer input_A(context, CL_MEM_READ_ONLY|CL_CHANNEL_1_INTELFPGA, N * N*sizeof(double));
    cl::Buffer input_output_C(context, CL_MEM_READ_WRITE|CL_CHANNEL_3_INTELFPGA, N * N*sizeof(double));

    std::string kernels[]={"test_dsyrk_notrans_lower","test_dsyrk_notrans_upper","test_dsyrk_trans_lower","test_dsyrk_trans_upper"};

    for(int in=0; in<nd;in++)  //N Dimension loop
    {
        int n=ndim[in];

        for(int ik=0; ik<nd;ik++)  //K Dimension loop
        {
            int k=ndim[ik];

            for(int icta=0; icta<2;icta++)  //A notrans/trans
            {
                bool transposedA = (icht[icta]=='T');
                FblasTranspose transA = (icht[icta]=='T')? FBLAS_TRANSPOSED : FBLAS_NO_TRANSPOSED;

                //generate A
                int na=(transposedA)?k:n;
                int ka=(transposedA)?n:k;
                int lda=(transposedA)?n:k;

                generate_matrix<double>(A,na,ka);

                //copy to device
                queue.enqueueWriteBuffer(input_A,CL_TRUE,0,na*ka*sizeof(double),A);


                for(int icu=0; icu<2;icu++)  //C lower/upper
                {

                    for(int ia = 0; ia < nalf ; ia++) //loop over alphas
                    {
                        double alpha=alphas[ia];

                        for(int ib = 0; ib < nbeta ; ib++) //loop over betas
                        {
                            double beta=betas[ib];
                            bool lowerC = (ichu[icu]=='L');
                            FblasUpLo uploC=(ichu[icu]=='L')?FBLAS_LOWER:FBLAS_UPPER;

                            //generate C
                            generate_matrix<double>(C,n,n);

                            //generate C
                            queue.enqueueWriteBuffer(input_output_C,CL_TRUE,0,n*n*sizeof(double),C);

                            fb.dsyrk(kernels[icta*2+icu],uploC,transA,n,k,alpha,input_A,lda,beta,input_output_C,n);

                            //copy back
                            queue.enqueueReadBuffer(input_output_C,CL_TRUE,0,n*n*sizeof(double),fpga_res);

                            check_result<double>(transposedA,lowerC,n,k,alpha,A,lda,beta,C,n,fpga_res);
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
void check_result (bool trans, bool lowerC, unsigned int N, unsigned int K,
                   T alpha,  T *A, unsigned int lda, T beta, T* C, unsigned int ldc, T *fpga_res)
{

    if (beta == 0.0) {
        if (!lowerC) {
            for (int i = 0; i < N; i++) {
                for (int j = i; j < N; j++) {
                    C[ldc * i + j] = 0.0;
                }
            }
        } else {
            for (int i = 0; i < N; i++) {
                for (int j = 0; j <= i; j++) {
                    C[ldc * i + j] = 0.0;
                }
            }
        }
    } else if (beta != 1.0) {
        if (!lowerC) {
            for (int i = 0; i < N; i++) {
                for (int j = i; j < N; j++) {
                    C[ldc * i + j] *= beta;
                }
            }
        } else {
            for (int i = 0; i < N; i++) {
                for (int j = 0; j <= i; j++) {
                    C[ldc * i + j] *= beta;
                }
            }
        }
    }

    if(!trans && lowerC)
    {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j <= i; j++) {
                T temp = 0.0;
                for (int k = 0; k < K; k++) {
                    temp += A[i * lda + k] * A[j * lda + k];
                }
                C[i * ldc + j] += alpha * temp;
            }
        }
    }
    if(!trans && !lowerC)
    {
        for (int i = 0; i < N; i++) {
            for (int j = i; j < N; j++) {
                T temp = 0.0;
                for (int k = 0; k < K; k++) {
                    temp += A[i * lda + k] * A[j * lda + k];
                }
                C[i * ldc + j] += alpha * temp;
            }
        }
    }
    if(trans && lowerC)
    {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j <= i; j++) {
                T temp = 0.0;
                for (int k = 0; k < K; k++) {
                    temp += A[k * lda + i] * A[k * lda + j];
                }
                C[i * ldc + j] += alpha * temp;
            }
        }
    }
    if(trans && !lowerC)
    {
        for (int i = 0; i < N; i++) {
            for (int j = i; j < N; j++) {
                T temp = 0.0;
                for (int k = 0; k < K; k++) {
                    temp += A[k * lda + i] * A[k * lda + j];
                }
                C[i * ldc + j] += alpha * temp;
            }
        }
    }
CHECK:
    //print_float_matrix(fpga_res,NN,MM);
    //    print_float_matrix(C,NN,MM);
    //    Measure error by considering nrm_inf
    T nrminf_diff=0, nrminf_orig=0;
    T error;

    for(int i=0;i < N; i++)
    {
        T nrminf=0, nrminf_o=0;
        for(int j=0; j<N;j++)
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
