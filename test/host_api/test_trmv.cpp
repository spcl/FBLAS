/**
<<<<<<< HEAD
  Tests for TRMV routine.
  Tests ideas borrowed from Blas testing and GSL (Gnu Scientific Library) v2.5
=======
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2020 ETH-Zurich. All rights reserved.
    See LICENSE for license information.
    Tests for TRMV routine.
    Tests ideas borrowed from Blas testing and GSL (Gnu Scientific Library) v2.5
>>>>>>> master
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


FBLASEnvironment fb;

template <typename T>
void check_result (bool trans, bool lower, int n, T *A, int lda, T *x, int incx, T *fpga_res);

TEST(TestSTrmv,TestStrmv)
{
    cl::CommandQueue queue;
    cl::Context context=fb.get_context();
    cl::Device device=fb.get_device();
    IntelFPGAOCLUtils::createCommandQueue(context,device,queue);
    int incx=1,incy=1;
    float *A,*x,*fpga_res;
    posix_memalign ((void **)&A, IntelFPGAOCLUtils::AOCL_ALIGNMENT, N*N*sizeof(float));
    posix_memalign ((void **)&x, IntelFPGAOCLUtils::AOCL_ALIGNMENT, N*max_inc*sizeof(float));
    posix_memalign ((void **)&fpga_res, IntelFPGAOCLUtils::AOCL_ALIGNMENT, N*max_inc*sizeof(float));

    cl::Buffer input_A(context, CL_MEM_READ_ONLY|CL_CHANNEL_1_INTELFPGA, N * N*sizeof(float));
    cl::Buffer input_x(context, CL_MEM_READ_WRITE|CL_CHANNEL_2_INTELFPGA, N * max_inc* sizeof(float));

    //std::string kernel_names[]={"test_sgemv_0","test_sgemv_1","test_sgemv_2","test_sgemv_3"};
    std::string kernel_names[]={"test_strmv_0","test_strmv_1","test_strmv_2","test_strmv_3","test_strmv_4","test_strmv_5","test_strmv_6","test_strmv_7"};
    std::string transposed_kernel_names[]={"test_strmv_trans_0","test_strmv_trans_1","test_strmv_trans_2","test_strmv_trans_3","test_strmv_trans_4","test_strmv_trans_5","test_strmv_trans_6","test_strmv_trans_7"};

    int test_case=1;
    for(int in=0; in<nd;++in)  //Dimension loop
    {
        int n=ndim[in];
        for(int icu=0; icu<2;icu++)  //C lower/upper
        {
            bool lower = (ichu[icu]=='L');
            FblasUpLo uplo=(ichu[icu]=='L')?FBLAS_LOWER:FBLAS_UPPER;
            for(int icta=0; icta<2;icta++)  //A notrans/trans
            {
                bool transposed = (icht[icta]=='T');
                FblasTranspose trans = (icht[icta]=='T')? FBLAS_TRANSPOSED : FBLAS_NO_TRANSPOSED;


                //Generate the matrix A
                generate_matrix<float>(A,n,n);
                queue.enqueueWriteBuffer(input_A,CL_TRUE,0,n*n*sizeof(float),A);


                for(int ix = 0; ix < nincs; ix ++) //incx and incy
                {
                    incx=incxs[ix];
                    int lx = abs(incx) * n;
                    generate_vector<float>(x,lx);

                    queue.enqueueWriteBuffer(input_x,CL_TRUE,0,lx*sizeof(float),x);
                    if(!transposed)
                        fb.strmv(kernel_names[icu*nincs+ix],uplo,trans,FBLAS_DIAG_UNDEF,n,input_A,n,input_x,incx);
                    else
                        fb.strmv(transposed_kernel_names[icu*nincs+ix],uplo,trans,FBLAS_DIAG_UNDEF,n,input_A,n,input_x,incx);
                    queue.enqueueReadBuffer(input_x,CL_TRUE,0,lx*sizeof(float),fpga_res);
                    //check
                    check_result<float>(transposed, lower, n,A,n,x,incx,fpga_res);
                    test_case++;


                }


            }

        }
    }
    //buffers and other objects are automatically deleted at program exit
}

TEST(TestDTrmv,TestDtrmv)
{
    cl::CommandQueue queue;
    cl::Context context=fb.get_context();
    cl::Device device=fb.get_device();
    IntelFPGAOCLUtils::createCommandQueue(context,device,queue);
    int incx=1,incy=1;
    double *A,*x,*fpga_res;
    posix_memalign ((void **)&A, IntelFPGAOCLUtils::AOCL_ALIGNMENT, N*N*sizeof(double));
    posix_memalign ((void **)&x, IntelFPGAOCLUtils::AOCL_ALIGNMENT, N*max_inc*sizeof(double));
    posix_memalign ((void **)&fpga_res, IntelFPGAOCLUtils::AOCL_ALIGNMENT, N*max_inc*sizeof(double));

    cl::Buffer input_A(context, CL_MEM_READ_ONLY|CL_CHANNEL_1_INTELFPGA, N * N*sizeof(double));
    cl::Buffer input_x(context, CL_MEM_READ_WRITE|CL_CHANNEL_2_INTELFPGA, N * max_inc* sizeof(double));

    //std::string kernel_names[]={"test_sgemv_0","test_sgemv_1","test_sgemv_2","test_sgemv_3"};
    std::string kernel_names[]={"test_dtrmv_0","test_dtrmv_1","test_dtrmv_2","test_dtrmv_3","test_dtrmv_4","test_dtrmv_5","test_dtrmv_6","test_dtrmv_7"};
    std::string transposed_kernel_names[]={"test_dtrmv_trans_0","test_dtrmv_trans_1","test_dtrmv_trans_2","test_dtrmv_trans_3","test_dtrmv_trans_4","test_dtrmv_trans_5","test_dtrmv_trans_6","test_dtrmv_trans_7"};

    int test_case=1;
    for(int in=0; in<nd;++in)  //Dimension loop
    {
        int n=ndim[in];
        for(int icu=0; icu<2;icu++)  //C lower/upper
        {
            bool lower = (ichu[icu]=='L');
            FblasUpLo uplo=(ichu[icu]=='L')?FBLAS_LOWER:FBLAS_UPPER;
            for(int icta=0; icta<2;icta++)  //A notrans/trans
            {
                bool transposed = (icht[icta]=='T');
                FblasTranspose trans = (icht[icta]=='T')? FBLAS_TRANSPOSED : FBLAS_NO_TRANSPOSED;


                //Generate the matrix A
                generate_matrix<double>(A,n,n);
                queue.enqueueWriteBuffer(input_A,CL_TRUE,0,n*n*sizeof(double),A);


                for(int ix = 0; ix < nincs; ix ++) //incx and incy
                {
                    incx=incxs[ix];
                    int lx = abs(incx) * n;
                    generate_vector<double>(x,lx);

                    queue.enqueueWriteBuffer(input_x,CL_TRUE,0,lx*sizeof(double),x);
                    if(!transposed)
                        fb.dtrmv(kernel_names[icu*nincs+ix],uplo,trans,FBLAS_DIAG_UNDEF,n,input_A,n,input_x,incx);
                    else
                        fb.dtrmv(transposed_kernel_names[icu*nincs+ix],uplo,trans,FBLAS_DIAG_UNDEF,n,input_A,n,input_x,incx);
                    queue.enqueueReadBuffer(input_x,CL_TRUE,0,lx*sizeof(double),fpga_res);
                    //check
                    check_result<double>(transposed, lower, n,A,n,x,incx,fpga_res);
                    test_case++;


                }


            }

        }
    }
    //buffers and other objects are automatically deleted at program exit
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
void check_result (bool trans, bool lower, int n, T *A, int lda, T *x, int incx, T *fpga_res)
{
    if(!trans && lower)
    {
        int ix = OFFSET(n, incx) + (n - 1) * incx;
        for (int i = n; i > 0 && i--;) {
            T temp = 0.0;
            const int j_min = 0;
            const int j_max = i;
            int jx = OFFSET(n, incx) + j_min * incx;
            for (int j = j_min; j < j_max; j++) {
                temp += x[jx] * A[lda * i + j];
                jx += incx;
            }
            x[ix] = temp + x[ix] * A[lda * i + i];
            ix -= incx;
        }
    }

    if(!trans && !lower)
    {
        int ix = OFFSET(n, incx);
        for (int i = 0; i < n; i++) {
            T temp = 0.0;
            const int j_min = i + 1;
            const int j_max = n;
            int jx = OFFSET(n, incx) + j_min * incx;
            for (int j = j_min; j < j_max; j++) {
                temp += x[jx] * A[lda * i + j];
                jx += incx;
            }
            x[ix] = temp + x[ix] * A[lda * i + i];
            ix += incx;
        }
    }

    if(trans && lower)
    {
        int ix = OFFSET(n, incx);
        for (int i = 0; i < n; i++) {
            T temp = 0.0;
            const int j_min = i + 1;
            const int j_max = n;
            int jx = OFFSET(n, incx) + (i + 1) * incx;
            for (int j = j_min; j < j_max; j++) {
                temp += x[jx] * A[lda * j + i];
                jx += incx;
            }
            x[ix] = temp + x[ix] * A[lda * i + i];
            ix += incx;
        }
    }
    if(trans && !lower)
    {
        int ix = OFFSET(n, incx) + (n - 1) * incx;
        for (int i = n; i > 0 && i--;) {
            T temp = 0.0;
            const int j_min = 0;
            const int j_max = i;
            int jx = OFFSET(n, incx) + j_min * incx;
            for (int j = j_min; j < j_max; j++) {
                temp += x[jx] * A[lda * j + i];
                jx += incx;
            }
            x[ix] = temp + x[ix] * A[lda * i + i];
            ix -= incx;
        }
    }

CHECK:
    //check result using nrm1
    int ix = OFFSET(n, incx);
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
