/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2020 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    Tests for symv routine.
    Tests ideas borrowed from Blas testing and GSL (Gnu Scientific Library) v2.5
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
void check_result (bool lower, int n, T *A, int lda, T alpha, T *x, int incx, T beta, T *y, int incy, T *fpga_res);

TEST(TestSymv,TestSsymv)
{
    const int nalf=3;
    const int nbeta=3;
    const float alphas[nalf] = {0,1.0,0.7};
    const float betas[nbeta] = {0,1.0,0.9};

    cl::CommandQueue queue;
    cl::Context context=fb.get_context();
    cl::Device device=fb.get_device();
    IntelFPGAOCLUtils::createCommandQueue(context,device,queue);
    int incx=1,incy=1;
    float *A,*x,*y,*fpga_res;
    posix_memalign ((void **)&A, IntelFPGAOCLUtils::AOCL_ALIGNMENT, N*N*sizeof(float));
    posix_memalign ((void **)&x, IntelFPGAOCLUtils::AOCL_ALIGNMENT, N*max_inc*sizeof(float));
    posix_memalign ((void **)&y, IntelFPGAOCLUtils::AOCL_ALIGNMENT, M*max_inc*sizeof(float));
    posix_memalign ((void **)&fpga_res, IntelFPGAOCLUtils::AOCL_ALIGNMENT, N*max_inc*sizeof(float));

    cl::Buffer input_A(context, CL_MEM_READ_ONLY|CL_CHANNEL_1_INTELFPGA, N * N*sizeof(float));
    cl::Buffer input_x(context, CL_MEM_READ_WRITE|CL_CHANNEL_2_INTELFPGA, N * max_inc* sizeof(float));
    cl::Buffer input_y(context, CL_MEM_READ_WRITE|CL_CHANNEL_3_INTELFPGA, N * max_inc* sizeof(float));


    //std::string kernel_names[]={"test_sgemv_0","test_sgemv_1","test_sgemv_2","test_sgemv_3"};
    std::string kernel_names[]={"test_ssymv_0","test_ssymv_1","test_ssymv_2","test_ssymv_3","test_ssymv_4","test_ssymv_5","test_ssymv_6","test_ssymv_7"};

    for(int in=0; in<nd;++in)  //Dimension loop
    {
        int n=ndim[in];
        for(int icu=0; icu<2;icu++)  //C lower/upper
        {
            bool lower = (ichu[icu]=='L');
            FblasUpLo uplo=(ichu[icu]=='L')?FBLAS_LOWER:FBLAS_UPPER;

            //Generate the matrix A
            generate_matrix<float>(A,n,n);
            queue.enqueueWriteBuffer(input_A,CL_TRUE,0,n*n*sizeof(float),A);


            for(int ix = 0; ix < nincs; ix ++) //incx and incy
            {
                incx=incxs[ix];
                incy=incys[ix];
                int lx = abs(incx) * n;
                int ly = abs(incy) * n;

                //loops over alpha and beta
                for(int ia = 0; ia < nalf ; ia++)
                {
                    float alpha=alphas[ia];
                    for(int ib = 0; ib < nbeta; ib++ )
                    {
                        float beta = betas[ib];
                        //generate the vectors
                        generate_vector<float>(x,lx);
                        generate_vector<float>(y,ly);

                        queue.enqueueWriteBuffer(input_x,CL_TRUE,0,lx*sizeof(float),x);
                        queue.enqueueWriteBuffer(input_y,CL_TRUE,0,ly*sizeof(float),y);
                        fb.ssymv(kernel_names[icu*nincs+ix],uplo,n,alpha,input_A,n,input_x,incx,beta,input_y,incy);
                        queue.enqueueReadBuffer(input_y,CL_TRUE,0,ly*sizeof(float),fpga_res);
                        //check
                        check_result<float>(lower,n,A,n,alpha,x,incx,beta,y,incy,fpga_res);

                    }
                }
            }
        }
    }
    //buffers and other objects are automatically deleted at program exit
}

TEST(TestDymv,TestDsymv)
{
    const int nalf=3;
    const int nbeta=3;
    const double alphas[nalf] = {0,1.0,0.7};
    const double betas[nbeta] = {0,1.0,0.9};

    cl::CommandQueue queue;
    cl::Context context=fb.get_context();
    cl::Device device=fb.get_device();
    IntelFPGAOCLUtils::createCommandQueue(context,device,queue);
    int incx=1,incy=1;
    double *A,*x,*y,*fpga_res;
    posix_memalign ((void **)&A, IntelFPGAOCLUtils::AOCL_ALIGNMENT, N*N*sizeof(double));
    posix_memalign ((void **)&x, IntelFPGAOCLUtils::AOCL_ALIGNMENT, N*max_inc*sizeof(double));
    posix_memalign ((void **)&y, IntelFPGAOCLUtils::AOCL_ALIGNMENT, M*max_inc*sizeof(double));
    posix_memalign ((void **)&fpga_res, IntelFPGAOCLUtils::AOCL_ALIGNMENT, N*max_inc*sizeof(double));

    cl::Buffer input_A(context, CL_MEM_READ_ONLY|CL_CHANNEL_1_INTELFPGA, N * N*sizeof(double));
    cl::Buffer input_x(context, CL_MEM_READ_WRITE|CL_CHANNEL_2_INTELFPGA, N * max_inc* sizeof(double));
    cl::Buffer input_y(context, CL_MEM_READ_WRITE|CL_CHANNEL_3_INTELFPGA, N * max_inc* sizeof(double));


    //std::string kernel_names[]={"test_sgemv_0","test_sgemv_1","test_sgemv_2","test_sgemv_3"};
    std::string kernel_names[]={"test_dsymv_0","test_dsymv_1","test_dsymv_2","test_dsymv_3","test_dsymv_4","test_dsymv_5","test_dsymv_6","test_dsymv_7"};

    for(int in=0; in<nd;++in)  //Dimension loop
    {
        int n=ndim[in];
        for(int icu=0; icu<2;icu++)  //C lower/upper
        {
            bool lower = (ichu[icu]=='L');
            FblasUpLo uplo=(ichu[icu]=='L')?FBLAS_LOWER:FBLAS_UPPER;

            //Generate the matrix A
            generate_matrix<double>(A,n,n);
            queue.enqueueWriteBuffer(input_A,CL_TRUE,0,n*n*sizeof(double),A);


            for(int ix = 0; ix < nincs; ix ++) //incx and incy
            {
                incx=incxs[ix];
                incy=incys[ix];
                int lx = abs(incx) * n;
                int ly = abs(incy) * n;

                //loops over alpha and beta
                for(int ia = 0; ia < nalf ; ia++)
                {
                    double alpha=alphas[ia];
                    for(int ib = 0; ib < nbeta; ib++ )
                    {
                        double beta = betas[ib];
                        //generate the vectors
                        generate_vector<double>(x,lx);
                        generate_vector<double>(y,ly);

                        queue.enqueueWriteBuffer(input_x,CL_TRUE,0,lx*sizeof(double),x);
                        queue.enqueueWriteBuffer(input_y,CL_TRUE,0,ly*sizeof(double),y);
                        fb.dsymv(kernel_names[icu*nincs+ix],uplo,n,alpha,input_A,n,input_x,incx,beta,input_y,incy);
                        queue.enqueueReadBuffer(input_y,CL_TRUE,0,ly*sizeof(double),fpga_res);
                        //check
                        check_result<double>(lower,n,A,n,alpha,x,incx,beta,y,incy,fpga_res);

                    }
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
void check_result (bool lower, int n, T *A, int lda, T alpha, T *x, int incx, T beta, T *y, int incy, T *fpga_res)
{
    /* print_float_matrix(A,n,n);
    print_float_vector(x,n);
    print_float_vector(y,n);
    std::cout << "n " << n << "lda " << lda << "incx "<<incx << "incy "<<incy <<std::endl;
    std::cout << "Alpha " << alpha << " Beta " <<beta <<std::endl;
*/
    int iy = OFFSET(n, incy);
    for (int i = 0; i< n;i++)
    {
        y[iy]*=beta;
        iy+=incy;
    }

    if(lower)
    {
        int ix = OFFSET(n, incx) + (n - 1) * incx;
        iy = OFFSET(n, incy) + (n - 1) * incy;
        for (int i = n; i > 0 && i--;) {
            T temp1 = alpha * x[ix];
            T temp2 = 0.0;
            const int j_min = 0;
            const int j_max = i;
            int jx = OFFSET(n, incx) + j_min * incx;
            int jy = OFFSET(n, incy) + j_min * incy;
            y[iy] += temp1 * A[lda * i + i];
            for (int j = j_min; j < j_max; j++) {
                y[jy] += temp1 * A[lda * i + j];
                temp2 += x[jx] * A[lda * i + j];
                jx += incx;
                jy += incy;
            }
            y[iy] += alpha * temp2;
            ix -= incx;
            iy -= incy;
        }
    }
    else
    {
        int ix = OFFSET(n, incx);
        int iy = OFFSET(n, incy);
        for (int i = 0; i < n; i++) {
            float temp1 = alpha * x[ix];
            float temp2 = 0.0;
            const int j_min = i + 1;
            const int j_max = n;
            int jx = OFFSET(n, incx) + j_min * incx;
            int jy = OFFSET(n, incy) + j_min * incy;
            y[iy] += temp1 * A[lda * i + i];
            for (int j = j_min; j < j_max; j++) {
                y[jy] += temp1 * A[lda * i + j];
                temp2 += x[jx] * A[lda * i + j];
                jx += incx;
                jy += incy;
            }
            y[iy] += alpha * temp2;
            ix += incx;
            iy += incy;
        }
    }

CHECK:
    //check result using nrm1
    iy = OFFSET(n, incy);
    //Measure error by considering
    T nrm1_diff=0, nrm1_orig=0;
    T error;
    for(int i=0;i < n; i++)
    {
        nrm1_diff+=abs(fpga_res[iy]-y[iy]);
        nrm1_orig+=abs(y[iy]);
        iy+=incy;
    }
    if(nrm1_diff == 0  && nrm1_orig == 0 )
        error=0;
    else
        error=nrm1_diff/nrm1_orig;
    if (std::is_same<T, float>::value)
        ASSERT_LE(error,flteps);
    else
        ASSERT_LE(error,dbleps);
}
