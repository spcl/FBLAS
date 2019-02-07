/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2019 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

  Tests for SYR routine.
  Tests ideas borrowed from BLAS testing and GSL (Gnu Scientific Library) v2.5
*/
#include <gtest/gtest.h>
#include <string>
#include <exception>
#include <algorithm>
#include <string.h>
#include "../../include/utils/ocl_utils.hpp"
#include "../../include/fblas_environment.hpp"
#include "test_tier2.hpp"

FBLASEnvironment fb;
const int nalf=3;               //number of different alpha values

template <typename T>
void check_result (bool lower, int n, T alpha, T *x, int incx, T *A, int lda, T *fpga_res);

TEST(TestSSyr,TestSsyr)
{
    const float alphas[nalf] = {0,1.0,0.7};
    cl::CommandQueue queue;
    cl::Context context=fb.get_context();
    cl::Device device=fb.get_device();
    IntelFPGAOCLUtils::createCommandQueue(context,device,queue);
    int incx=1,incy=1;
    float *A,*x,*fpga_res;
    posix_memalign ((void **)&A, IntelFPGAOCLUtils::AOCL_ALIGNMENT, N*N*sizeof(float));
    posix_memalign ((void **)&x, IntelFPGAOCLUtils::AOCL_ALIGNMENT, N*max_inc*sizeof(float));
    posix_memalign ((void **)&fpga_res, IntelFPGAOCLUtils::AOCL_ALIGNMENT, N*N*sizeof(float));

    cl::Buffer input_A(context, CL_MEM_READ_WRITE|CL_CHANNEL_1_INTELFPGA, N * N*sizeof(float));
    cl::Buffer input_x(context, CL_MEM_READ_ONLY|CL_CHANNEL_2_INTELFPGA, N * max_inc* sizeof(float));

    std::string kernel_names[]={"test_ssyr_0","test_ssyr_1","test_ssyr_2","test_ssyr_3","test_ssyr_4","test_ssyr_5","test_ssyr_6","test_ssyr_7"};
    int test_case=1;
    for(int in=0; in<nd;++in)  //Dimension loop
    {
        int n=ndim[in];
        for(int icu=0; icu<2;icu++)  //A lower/upper
        {
            bool lower = (ichu[icu]=='L');
            FblasUpLo uplo=(ichu[icu]=='L')?FBLAS_LOWER:FBLAS_UPPER;
            for(int ix = 0; ix < nincs; ix ++) //incx and incy
            {
                incx=incxs[ix];
                int lx = abs(incx) * n;

                //loops over alpha
                for(int ia = 0; ia < nalf ; ia++)
                {
                    float alpha=alphas[ia];
                    //generate the vectors
                    generate_vector<float>(x,lx);
                    //Generate the matrix A (entirely to be sure that the other part is not touched)
                    generate_matrix<float>(A,n,n);

                    //copy everything to device
                    queue.enqueueWriteBuffer(input_x,CL_TRUE,0,lx*sizeof(float),x);
                    queue.enqueueWriteBuffer(input_A,CL_TRUE,0,n*n*sizeof(float),A);
                    fb.ssyr(kernel_names[icu*nincs+ix],uplo,n,alpha,input_x,incx,input_A,n);

                    queue.enqueueReadBuffer(input_A,CL_TRUE,0,n*n*sizeof(float),fpga_res);
                    //check
                    check_result<float>(lower,n,alpha,x,incx,A,n,fpga_res);
                    test_case++;
                }
            }
        }
    }
}
TEST(TestDSyr,TestDsyr)
{
    const double alphas[nalf] = {0,1.0,0.7};
    cl::CommandQueue queue;
    cl::Context context=fb.get_context();
    cl::Device device=fb.get_device();
    IntelFPGAOCLUtils::createCommandQueue(context,device,queue);
    int incx=1,incy=1;
    double *A,*x,*fpga_res;
    posix_memalign ((void **)&A, IntelFPGAOCLUtils::AOCL_ALIGNMENT, N*N*sizeof(double));
    posix_memalign ((void **)&x, IntelFPGAOCLUtils::AOCL_ALIGNMENT, N*max_inc*sizeof(double));
    posix_memalign ((void **)&fpga_res, IntelFPGAOCLUtils::AOCL_ALIGNMENT, N*N*sizeof(double));

    cl::Buffer input_A(context, CL_MEM_READ_WRITE|CL_CHANNEL_1_INTELFPGA, N * N*sizeof(double));
    cl::Buffer input_x(context, CL_MEM_READ_ONLY|CL_CHANNEL_2_INTELFPGA, N * max_inc* sizeof(double));

    std::string kernel_names[]={"test_dsyr_0","test_dsyr_1","test_dsyr_2","test_dsyr_3","test_dsyr_4","test_dsyr_5","test_dsyr_6","test_dsyr_7"};
    int test_case=1;
    for(int in=0; in<nd;++in)  //Dimension loop
    {
        int n=ndim[in];
        for(int icu=0; icu<2;icu++)  //A lower/upper
        {
            bool lower = (ichu[icu]=='L');
            FblasUpLo uplo=(ichu[icu]=='L')?FBLAS_LOWER:FBLAS_UPPER;
            for(int ix = 0; ix < nincs; ix ++) //incx and incy
            {
                incx=incxs[ix];
                int lx = abs(incx) * n;

                //loops over alpha
                for(int ia = 0; ia < nalf ; ia++)
                {
                    double alpha=alphas[ia];
                    //generate the vectors
                    generate_vector<double>(x,lx);
                    //Generate the matrix A (entirely to be sure that the other part is not touched)
                    generate_matrix<double>(A,n,n);

                    //copy everything to device
                    queue.enqueueWriteBuffer(input_x,CL_TRUE,0,lx*sizeof(double),x);
                    queue.enqueueWriteBuffer(input_A,CL_TRUE,0,n*n*sizeof(double),A);
                    fb.dsyr(kernel_names[icu*nincs+ix],uplo,n,alpha,input_x,incx,input_A,n);

                    queue.enqueueReadBuffer(input_A,CL_TRUE,0,n*n*sizeof(double),fpga_res);
                    //check
                    check_result<double>(lower,n,alpha,x,incx,A,n,fpga_res);
                    test_case++;
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
void check_result (bool lower, int n, T alpha, T *x, int incx, T *A, int lda, T *fpga_res)
{

    if(!lower)
    {

        int ix = OFFSET(n, incx);
        for (int i = 0; i < n; i++) {
            const T tmp = alpha * x[ix];
            int jx = ix;
            for (int j = i; j < n; j++) {
                A[lda * i + j] += x[jx] * tmp;
                jx += incx;
            }
            ix += incx;
        }
    }
    else
    {
        int ix = OFFSET(n, incx);
        for (int i = 0; i < n; i++) {
            const T tmp = alpha * x[ix];
            int jx = OFFSET(n, incx);
            for (int j = 0; j <= i; j++) {
                A[lda * i + j] += x[jx] * tmp;
                jx += incx;
            }
            ix += incx;
        }
    }

CHECK:
    //check result
    //compute nrm inf
    T nrminf_diff=0, nrminf_orig=0;
    T error;


    for(int i=0;i < n; i++)
    {
        T nrminf=0, nrminf_o=0;
        for(int j=0; j<n;j++)
        {
            nrminf+=abs(fpga_res[i*lda+j]-A[i*lda+j]);
            nrminf_o+=abs(A[i*lda+j]);
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


