/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2020 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    Tests for TRSM routine.
    Tests ideas borrowed from BLAS testing
    TRSM check routine is a modified version of the one included in GSL (Gnu Scientific Library) v2.5
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
void check_result (bool left_side, bool lower, bool transposed, unsigned int n, unsigned int m, T alpha,  T *A, unsigned int lda, T *B, unsigned int ldb, T *fpga_res);

FBLASEnvironment fb;
const int N=64;                 // max N size
const int nd=3;
const int ndim[3]={4,16,64};    //sizes



TEST(TestSTrsv,TestStrsv)
{
    cl::CommandQueue queue;
    cl::Context context=fb.get_context();
    cl::Device device=fb.get_device();
    IntelFPGAOCLUtils::createCommandQueue(context,device,queue);
    float *A,*B,*fpga_res;
    const int nalf=3;
    const float alphas[nalf] = {0,1.0,0.7};
    posix_memalign ((void **)&A, IntelFPGAOCLUtils::AOCL_ALIGNMENT, N*N*sizeof(float));
    posix_memalign ((void **)&B, IntelFPGAOCLUtils::AOCL_ALIGNMENT, N*N*sizeof(float));
    posix_memalign ((void **)&fpga_res, IntelFPGAOCLUtils::AOCL_ALIGNMENT, N*N*sizeof(float));

    cl::Buffer input_A(context, CL_MEM_READ_ONLY|CL_CHANNEL_1_INTELFPGA, N * N*sizeof(float));
    cl::Buffer input_output_B(context, CL_MEM_READ_ONLY|CL_CHANNEL_2_INTELFPGA, N * N*sizeof(float));

    std::string left_side_kernels[]={"test_strsm_left_lower_notrans","test_strsm_left_lower_trans","test_strsm_left_upper_notrans","test_strsm_left_upper_trans"};
    std::string right_side_kernels[]={"test_strsm_right_lower_notrans","test_strsm_right_lower_trans","test_strsm_right_upper_notrans","test_strsm_right_upper_trans"};


    for(int in=0; in<nd;++in)  //N Dimension loop
    {
        int n=ndim[in];

        for(int im=0; im<nd;++im)  //M Dimension loop
        {
            int m=ndim[im];

            for(int ics=0; ics <2 ;ics++) //Side loop
            {
                FblasSide side=(ichs[ics]=='L')?FBLAS_LEFT:FBLAS_RIGHT;
                bool left_side=(ichs[ics]=='L');
                int lda=(left_side)?n:m;
                for(int icu=0; icu<2;icu++) //lower/upper
                {
                    bool lower = (ichu[icu]=='L');
                    FblasUpLo uplo= (ichu[icu]=='L')? FBLAS_LOWER : FBLAS_UPPER;
                    for(int ict=0; ict<2;ict++) //notrans/trans
                    {
                        bool transposed = (icht[ict]=='T');
                        FblasTranspose trans = (icht[ict]=='T')? FBLAS_TRANSPOSED : FBLAS_NO_TRANSPOSED;

                        for(int ia = 0; ia < nalf ; ia++) //loop over alpha
                        {
                            float alpha=alphas[ia];
                            //generate the matrices
                            generate_matrix<float>(A,lda,lda);
                            generate_matrix<float>(B,n,m);

                            //copy everything to device
                            queue.enqueueWriteBuffer(input_A,CL_TRUE,0,lda*lda*sizeof(float),A);
                            queue.enqueueWriteBuffer(input_output_B,CL_TRUE,0,n*m*sizeof(float),B);

                            if(ics==0)
                                fb.strsm(left_side_kernels[icu*2+ict],side,trans,uplo,n,m,alpha,input_A,lda,input_output_B,m);
                            else
                                fb.strsm(right_side_kernels[icu*2+ict],side,trans,uplo,n,m,alpha,input_A,lda,input_output_B,m);

                            //copy back
                            queue.enqueueReadBuffer(input_output_B,CL_TRUE,0,n*m*sizeof(float),fpga_res);
                            //check
                            check_result<float>(left_side,lower,transposed,n,m,alpha,A,lda,B,m,fpga_res);


                        }
                    }


                }
            }
        }
    }
}

TEST(TestDTrsv,TestDtrsv)
{
    cl::CommandQueue queue;
    cl::Context context=fb.get_context();
    cl::Device device=fb.get_device();
    IntelFPGAOCLUtils::createCommandQueue(context,device,queue);
    double *A,*B,*fpga_res;
    const int nalf=3;
    const double alphas[nalf] = {0,1.0,0.7};
    posix_memalign ((void **)&A, IntelFPGAOCLUtils::AOCL_ALIGNMENT, N*N*sizeof(double));
    posix_memalign ((void **)&B, IntelFPGAOCLUtils::AOCL_ALIGNMENT, N*N*sizeof(double));
    posix_memalign ((void **)&fpga_res, IntelFPGAOCLUtils::AOCL_ALIGNMENT, N*N*sizeof(double));

    cl::Buffer input_A(context, CL_MEM_READ_ONLY|CL_CHANNEL_1_INTELFPGA, N * N*sizeof(double));
    cl::Buffer input_output_B(context, CL_MEM_READ_ONLY|CL_CHANNEL_2_INTELFPGA, N * N*sizeof(double));

    std::string left_side_kernels[]={"test_dtrsm_left_lower_notrans","test_dtrsm_left_lower_trans","test_dtrsm_left_upper_notrans","test_dtrsm_left_upper_trans"};
    std::string right_side_kernels[]={"test_dtrsm_right_lower_notrans","test_dtrsm_right_lower_trans","test_dtrsm_right_upper_notrans","test_dtrsm_right_upper_trans"};


    for(int in=0; in<nd;++in)  //N Dimension loop
    {
        int n=ndim[in];

        for(int im=0; im<nd;++im)  //M Dimension loop
        {
            int m=ndim[im];

            for(int ics=0; ics <2 ;ics++) //Side loop
            {
                FblasSide side=(ichs[ics]=='L')?FBLAS_LEFT:FBLAS_RIGHT;
                bool left_side=(ichs[ics]=='L');
                int lda=(left_side)?n:m;
                for(int icu=0; icu<2;icu++) //lower/upper
                {
                    bool lower = (ichu[icu]=='L');
                    FblasUpLo uplo= (ichu[icu]=='L')? FBLAS_LOWER : FBLAS_UPPER;
                    for(int ict=0; ict<2;ict++) //notrans/trans
                    {
                        bool transposed = (icht[ict]=='T');
                        FblasTranspose trans = (icht[ict]=='T')? FBLAS_TRANSPOSED : FBLAS_NO_TRANSPOSED;

                        for(int ia = 0; ia < nalf ; ia++) //loop over alpha
                        {
                            double alpha=alphas[ia];
                            //generate the matrices
                            generate_matrix<double>(A,lda,lda);
                            generate_matrix<double>(B,n,m);

                            //copy everything to device
                            queue.enqueueWriteBuffer(input_A,CL_TRUE,0,lda*lda*sizeof(double),A);
                            queue.enqueueWriteBuffer(input_output_B,CL_TRUE,0,n*m*sizeof(double),B);
                            if(ics==0)
                                fb.dtrsm(left_side_kernels[icu*2+ict],side,trans,uplo,n,m,alpha,input_A,lda,input_output_B,m);
                            else
                                fb.dtrsm(right_side_kernels[icu*2+ict],side,trans,uplo,n,m,alpha,input_A,lda,input_output_B,m);

                            //copy back
                            queue.enqueueReadBuffer(input_output_B,CL_TRUE,0,n*m*sizeof(double),fpga_res);
                            //check
                            check_result<double>(left_side,lower,transposed,n,m,alpha,A,lda,B,m,fpga_res);


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
void check_result (bool left_side, bool lower, bool transposed, unsigned int n, unsigned int m, T alpha,  T *A, unsigned int lda, T *B, unsigned int ldb, T *fpga_res)
{
    int NN=n;
    int MM=m;
    if(left_side && lower && !transposed) //Left, lower no transposed
    {
        if (alpha != 1.0) {
            for (int i = 0; i < NN; i++) {
                for (int j = 0; j < MM; j++) {
                    B[ldb * i + j] *= alpha;
                }
            }
        }

        for (int i = 0; i < NN; i++) {
            T Aii = A[lda * i + i];
            for (int j = 0; j < MM; j++) {
                B[ldb * i + j] /= Aii;
            }

            for (int k = i + 1; k < NN; k++) {
                T Aki = A[k * lda + i];
                for (int j = 0; j < MM; j++) {
                    B[ldb * k + j] -= Aki * B[ldb * i + j];
                }
            }
        }

    }
    if(left_side && lower && transposed) //left,lower and transposed
    {
        if (alpha != 1.0) {
            for (int i = 0; i < NN; i++) {
                for (int j = 0; j < MM; j++) {
                    B[ldb * i + j] *= alpha;
                }
            }
        }

        for (int i = NN; i > 0 && i--;) {
            T Aii = A[lda * i + i];
            for (int j = 0; j < MM; j++) {
                B[ldb * i + j] /= Aii;
            }

            for (int k = 0; k < i; k++) {
                const T Aik = A[i * lda + k];
                for (int j = 0; j < MM; j++) {
                    B[ldb * k + j] -= Aik * B[ldb * i + j];
                }
            }
        }
    }

    if(left_side && !lower && !transposed) //Left, upper no transposed
    {
        if (alpha != 1.0) {
            for (int i = 0; i < NN; i++) {
                for (int j = 0; j < MM; j++) {
                    B[ldb * i + j] *= alpha;
                }
            }
        }

        for (int i = NN; i > 0 && i--;) {
            T Aii = A[lda * i + i];
            for (int j = 0; j < MM; j++) {
                B[ldb * i + j] /= Aii;
            }

            for (int k = 0; k < i; k++) {
                T Aki = A[k * lda + i];
                for (int j = 0; j < MM; j++) {
                    B[ldb * k + j] -= Aki * B[ldb * i + j];
                }
            }
        }
    }

    if(left_side && !lower && transposed) //Left, upper and transposed
    {
        if (alpha != 1.0) {
            for (int i = 0; i < NN; i++) {
                for (int j = 0; j < MM; j++) {
                    B[ldb * i + j] *= alpha;
                }
            }
        }

        for (int i = 0; i < NN; i++) {
            T Aii = A[lda * i + i];
            for (int j = 0; j < MM; j++) {
                B[ldb * i + j] /= Aii;
            }

            for (int k = i + 1; k < NN; k++) {
                T Aik = A[i * lda + k];
                for (int j = 0; j < MM; j++) {
                    B[ldb * k + j] -= Aik * B[ldb * i + j];
                }
            }
        }
    }

    if(!left_side && lower && !transposed) //right, lower, non transposded
    {
        if (alpha != 1.0) {
            for (int i = 0; i < NN; i++) {
                for (int j = 0; j < MM; j++) {
                    B[ldb * i + j] *= alpha;
                }
            }
        }

        for (int i = 0; i < NN; i++) {
            for (int j = MM; j > 0 && j--;) {
                T Ajj = A[lda * j + j];
                B[ldb * i + j] /= Ajj;

                T Bij = B[ldb * i + j];
                for (int k = 0; k < j; k++) {
                    B[ldb * i + k] -= A[j * lda + k] * Bij;
                }
            }
        }
    }

    if(!left_side && lower && transposed) //Right, lower and transposed
    {

        if (alpha != 1.0) {
            for (int i = 0; i < NN; i++) {
                for (int j = 0; j < MM; j++) {
                    B[ldb * i + j] *= alpha;
                }
            }
        }

        for (int i = 0; i < NN; i++) {
            for (int j = 0; j < MM; j++) {
                T Ajj = A[lda * j + j];
                B[ldb * i + j] /= Ajj;

                T Bij = B[ldb * i + j];
                for (int k = j + 1; k < MM; k++) {
                    B[ldb * i + k] -= A[k * lda + j] * Bij;
                }

            }
        }

    }

    if(!left_side && !lower && !transposed) //Right, upper and non transposed
    {
        if (alpha != 1.0) {
            for (int i = 0; i < NN; i++) {
                for (int j = 0; j < MM; j++) {
                    B[ldb * i + j] *= alpha;
                }
            }
        }

        for (int i = 0; i < NN; i++) {
            for (int j = 0; j < MM; j++) {
                T Ajj = A[lda * j + j];
                B[ldb * i + j] /= Ajj;
                T Bij = B[ldb * i + j];
                for (int k = j + 1; k < MM; k++) {
                    B[ldb * i + k] -= A[j * lda + k] * Bij;
                }
            }
        }
    }

    if(!left_side && !lower && transposed) //Right, upper and transposed
    {
        if (alpha != 1.0) {
            for (int i = 0; i < NN; i++) {
                for (int j = 0; j < MM; j++) {
                    B[ldb * i + j] *= alpha;
                }
            }
        }

        for (int i = 0; i < NN; i++) {
            for (int j = MM; j > 0 && j--;) {

                T Ajj = A[lda * j + j];
                B[ldb * i + j] /= Ajj;
                T Bij = B[ldb * i + j];
                for (int k = 0; k < j; k++) {
                    B[ldb * i + k] -= A[k * lda + j] * Bij;
                }
            }
        }
    }

CHECK:
    //Measure error by considering nrm_inf
    T nrminf_diff=0, nrminf_orig=0;
    T error;

    for(int i=0;i < NN; i++)
    {
        T nrminf=0, nrminf_o=0;
        for(int j=0; j<MM;j++)
        {
            nrminf+=abs(fpga_res[i*ldb+j]-B[i*ldb+j]);
            nrminf_o+=abs(B[i*ldb+j]);
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
