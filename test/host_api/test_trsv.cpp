/**
<<<<<<< HEAD
  Tests for TRSV routine.
  Tests ideas borrowed from BLAS testing
  TRSV check routine is a modified version of the one included in GSL (Gnu Scientific Library) v2.5
=======
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2020 ETH-Zurich. All rights reserved.
    See LICENSE for license information.
    Tests for TRSV routine.
    Tests ideas borrowed from BLAS testing
    TRSV check routine is a modified version of the one included in GSL (Gnu Scientific Library) v2.5
>>>>>>> master
*/
#include <gtest/gtest.h>
#include <string>
#include <exception>
#include <algorithm>
#include <string.h>
#include "../../include/utils/ocl_utils.hpp"
#include "../../include/fblas_environment.hpp"
#include "test_tier2.hpp"
#define OFFSET(N, incx) ((incx) > 0 ?  0 : ((N) - 1) * (-(incx)))

FBLASEnvironment fb;


//g++ test_dot.cpp -lgtest $( aocl compile-config ) -std=c++11 $(aocl link-config) -lpthread -I/home/tdematt/lib/rapidjson/include

template <typename T>
void check_result (bool lower, bool transposed, int n, T *A, int lda, T *x, int incx,  T *fpga_res)
{

    if(n == 0)
        goto CHECK;
    int ix, jx;
    if(!transposed && !lower)
    {
        /* backsubstitution */
        ix = OFFSET(n, incx) + incx * (n - 1);
        x[ix] = x[ix] / A[lda * (n - 1) + (n - 1)];

        ix -= incx;
        for (int i = n - 1; i > 0 && i--;) {
            T tmp = x[ix];
            jx = ix + incx;
            for (int j = i + 1; j < n; j++) {
                const T Aij = A[lda * i + j];
                tmp -= Aij * x[jx];
                jx += incx;
            }
            x[ix] = tmp / A[lda * i + i];
            ix -= incx;
        }
    }
    else if (!transposed && lower){
        /* forward substitution */
        ix = OFFSET(n, incx);
        x[ix] = x[ix] / A[lda * 0 + 0];

        ix += incx;
        for (int i = 1; i < n; i++) {
            T tmp = x[ix];
            jx = OFFSET(n, incx);
            for (int j = 0; j < i; j++) {
                const T Aij = A[lda * i + j];
                tmp -= Aij * x[jx];
                jx += incx;
            }
            x[ix] = tmp / A[lda * i + i];
            ix += incx;
        }
    }
    else if (transposed && !lower)
    {
        /* form  x := inv( A' )*x */

        /* forward substitution */
        ix = OFFSET(n, incx);
        x[ix] = x[ix] / A[lda * 0 + 0];
        ix += incx;
        for (int i = 1; i < n; i++) {
            T tmp = x[ix];
            jx = OFFSET(n, incx);
            for (int j = 0; j < i; j++) {
                const T Aji = A[lda * j + i];
                tmp -= Aji * x[jx];
                jx += incx;
            }
            x[ix] = tmp / A[lda * i + i];
            ix += incx;
        }
    }
    else if (transposed && lower)
    {
        /* backsubstitution */
        ix = OFFSET(n, incx) + (n - 1) * incx;
        x[ix] = x[ix] / A[lda * (n - 1) + (n - 1)];
        ix -= incx;
        for (int i = n - 1; i > 0 && i--;) {
            T tmp = x[ix];
            jx = ix + incx;
            for (int j = i + 1; j < n; j++) {
                const T Aji = A[lda * j + i];
                tmp -= Aji * x[jx];
                jx += incx;
            }
            x[ix] = tmp / A[lda * i + i];
            ix -= incx;
        }


    }

CHECK:
    ix = OFFSET(n, incx);

    //check result using nrm1
    ix = OFFSET(n, incx);
    //Measure error by considering
    T nrm1_diff=0, nrm1_orig=0;
    T error;
    for(int i=0;i < n; i++)
    {
        nrm1_diff+=abs(fpga_res[ix]-x[ix]);
        nrm1_orig+=abs(x[ix]);
        ix+=incx;
    }
    error=nrm1_diff/nrm1_orig;
    if (std::is_same<T, float>::value)
        ASSERT_LE(error,flteps);
    else
        ASSERT_LE(error,dbleps);
}


TEST(TestSTrsv,TestStrsv)
{
    cl::CommandQueue queue;
    cl::Context context=fb.get_context();
    cl::Device device=fb.get_device();
    IntelFPGAOCLUtils::createCommandQueue(context,device,queue);
    int incx=1;
    float *A,*x,*fpga_res;
    posix_memalign ((void **)&A, IntelFPGAOCLUtils::AOCL_ALIGNMENT, N*N*sizeof(float));
    posix_memalign ((void **)&x, IntelFPGAOCLUtils::AOCL_ALIGNMENT, N*max_inc*sizeof(float));
    posix_memalign ((void **)&fpga_res, IntelFPGAOCLUtils::AOCL_ALIGNMENT, N*max_inc*sizeof(float));

    cl::Buffer input_A(context, CL_MEM_READ_ONLY|CL_CHANNEL_1_INTELFPGA, N * N*sizeof(float));
    cl::Buffer input_x(context, CL_MEM_READ_WRITE|CL_CHANNEL_2_INTELFPGA, N * max_inc* sizeof(float));

    std::string kernel_lower_names[2][nincs]={{"test_strsv_0","test_strsv_1","test_strsv_2","test_strsv_3"},
                                       {"test_strsv_trans_0","test_strsv_trans_1","test_strsv_trans_2","test_strsv_trans_3"}};
    std::string kernel_upper_names[2][nincs]={{"test_strsv_u_0","test_strsv_u_1","test_strsv_u_2","test_strsv_u_3"},
                                       {"test_strsv_trans_u_0","test_strsv_trans_u_1","test_strsv_trans_u_2","test_strsv_trans_u_3"}};
    int test_case=1;


    for(int in=0; in<nd;++in)  //Dimension loop
    {
        int n=ndim[in];
        //Generate the matrix A
        generate_matrix<float>(A,n,n);
        for(int icu=0; icu<2;icu++) //lower/upper
        {
            bool lower = (ichu[icu]=='L');
            FblasUpLo uplo= (ichu[icu]=='L')? FBLAS_LOWER : FBLAS_UPPER;
            for(int ict=0; ict<2;ict++) //notrans/trans
            {
                bool transposed = (icht[ict]=='T');
                FblasTranspose trans = (icht[ict]=='T')? FBLAS_TRANSPOSED : FBLAS_NO_TRANSPOSED;
                for(int ix = 0; ix < nincs; ix ++) //incx and incy
                {
                    incx=incxs[ix];
                    int lx = abs(incx) * n;


                    //generate the vectors
                    generate_vector<float>(x,lx);

                    //copy everything to device
                    queue.enqueueWriteBuffer(input_x,CL_TRUE,0,lx*sizeof(float),x);
                    queue.enqueueWriteBuffer(input_A,CL_TRUE,0,n*n*sizeof(float),A);

                    if(lower)
                        fb.strsv(kernel_lower_names[ict][ix],uplo,trans, n,input_A,n,input_x,incx);
                    else
                        fb.strsv(kernel_upper_names[ict][ix],uplo,trans, n,input_A,n,input_x,incx);
                    queue.enqueueReadBuffer(input_x,CL_TRUE,0,lx*sizeof(float),fpga_res);
                    //check
                    check_result<float>(lower,transposed,n,A,n,x,incx,fpga_res);
                    test_case++;

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
    int incx=1;
    double *A,*x,*fpga_res;
    posix_memalign ((void **)&A, IntelFPGAOCLUtils::AOCL_ALIGNMENT, N*N*sizeof(double));
    posix_memalign ((void **)&x, IntelFPGAOCLUtils::AOCL_ALIGNMENT, N*max_inc*sizeof(double));
    posix_memalign ((void **)&fpga_res, IntelFPGAOCLUtils::AOCL_ALIGNMENT, N*max_inc*sizeof(double));

    cl::Buffer input_A(context, CL_MEM_READ_ONLY|CL_CHANNEL_1_INTELFPGA, N * N*sizeof(double));
    cl::Buffer input_x(context, CL_MEM_READ_WRITE|CL_CHANNEL_2_INTELFPGA, N * max_inc* sizeof(double));

    std::string kernel_lower_names[2][nincs]={{"test_dtrsv_0","test_dtrsv_1","test_dtrsv_2","test_dtrsv_3"},
                                       {"test_dtrsv_trans_0","test_dtrsv_trans_1","test_dtrsv_trans_2","test_dtrsv_trans_3"}};
    std::string kernel_upper_names[2][nincs]={{"test_dtrsv_u_0","test_dtrsv_u_1","test_dtrsv_u_2","test_dtrsv_u_3"},
                                       {"test_dtrsv_trans_u_0","test_dtrsv_trans_u_1","test_dtrsv_trans_u_2","test_dtrsv_trans_u_3"}};
    int test_case=1;


    for(int in=0; in<nd;++in)  //Dimension loop
    {
        int n=ndim[in];
        //Generate the matrix A
        generate_matrix<double>(A,n,n);
        for(int icu=0; icu<2;icu++) //lower/upper
        {
            bool lower = (ichu[icu]=='L');
            FblasUpLo uplo= (ichu[icu]=='L')? FBLAS_LOWER : FBLAS_UPPER;
            for(int ict=0; ict<2;ict++) //notrans/trans
            {
                bool transposed = (icht[ict]=='T');
                FblasTranspose trans = (icht[ict]=='T')? FBLAS_TRANSPOSED : FBLAS_NO_TRANSPOSED;
                for(int ix = 0; ix < nincs; ix ++) //incx and incy
                {
                    incx=incxs[ix];
                    int lx = abs(incx) * n;


                    //generate the vectors
                    generate_vector<double>(x,lx);

                    //copy everything to device
                    queue.enqueueWriteBuffer(input_x,CL_TRUE,0,lx*sizeof(double),x);
                    queue.enqueueWriteBuffer(input_A,CL_TRUE,0,n*n*sizeof(double),A);
                    //
                    //std::cout << "Executing with n: " << n << " incx: "<<incx<<" incy: "<<incy<<" alpha: "<<alpha<<std::endl;
                   // std::cout << "Test lower: "<<lower<< " transposed: "<<icht[ict] <<" n: "<<n<<" incx: "<<incx << " lx: "<<lx <<std::endl;
                    if(lower)
                        fb.dtrsv(kernel_lower_names[ict][ix],uplo,trans, n,input_A,n,input_x,incx);
                    else
                        fb.dtrsv(kernel_upper_names[ict][ix],uplo,trans, n,input_A,n,input_x,incx);
                    queue.enqueueReadBuffer(input_x,CL_TRUE,0,lx*sizeof(double),fpga_res);
                    //check
                    check_result<double>(lower,transposed,n,A,n,x,incx,fpga_res);
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
