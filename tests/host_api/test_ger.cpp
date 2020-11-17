/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2020 ETH-Zurich. All rights reserved.
    See LICENSE for license information.
    Tests for GER routine.
    Tests ideas borrowed from BLAS testing
    GER check routine is a modified version of the one included in GSL (Gnu Scientific Library) v2.5
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
void check_result (int n, int m, T alpha, T *A, int lda, T *x, int incx, T *y, int incy, T *fpga_res);

TEST(TestSGer,TestSger)
{
    const float alphas[nalf] = {0,1.0,0.7};
    cl::CommandQueue queue;
    cl::Context context=fb.get_context();
    cl::Device device=fb.get_device();
    IntelFPGAOCLUtils::createCommandQueue(context,device,queue);
    int incx=1,incy=1;
    float *A,*x,*y,*fpga_res;
    posix_memalign ((void **)&A, IntelFPGAOCLUtils::AOCL_ALIGNMENT, N*M*sizeof(float));
    posix_memalign ((void **)&x, IntelFPGAOCLUtils::AOCL_ALIGNMENT, N*max_inc*sizeof(float));
    posix_memalign ((void **)&y, IntelFPGAOCLUtils::AOCL_ALIGNMENT, M*max_inc*sizeof(float));
    posix_memalign ((void **)&fpga_res, IntelFPGAOCLUtils::AOCL_ALIGNMENT, N*M*sizeof(float));

    cl::Buffer input_A(context, CL_MEM_READ_WRITE|CL_CHANNEL_1_INTELFPGA, N * M*sizeof(float));
    cl::Buffer input_x(context, CL_MEM_READ_ONLY|CL_CHANNEL_2_INTELFPGA, N * max_inc* sizeof(float));
    cl::Buffer input_y(context, CL_MEM_READ_WRITE|CL_CHANNEL_3_INTELFPGA, N * max_inc *sizeof(float));

    std::string kernel_names[]={"test_sger_0","test_sger_1","test_sger_2","test_sger_3"};
    int test_case=1;
    for(int in=0; in<nd;++in)  //Dimension loop
    {
        int n=ndim[in];
        int m=n/2+1;
        //Generate the matrix A
        generate_matrix<float>(A,n,m);


        for(int ix = 0; ix < nincs; ix ++) //incx and incy
        {
            incx=incxs[ix];
            incy=incys[ix];
            int lx = abs(incx) * n;
            int ly = abs(incy) * m;

            //loops over alpha
            for(int ia = 0; ia < nalf ; ia++)
            {
                float alpha=alphas[ia];
                //generate the vectors
                generate_vector<float>(x,lx);
                generate_vector<float>(y,ly);

                //copy everything to device
                queue.enqueueWriteBuffer(input_x,CL_TRUE,0,lx*sizeof(float),x);
                queue.enqueueWriteBuffer(input_y,CL_TRUE,0,ly*sizeof(float),y);
                queue.enqueueWriteBuffer(input_A,CL_TRUE,0,n*m*sizeof(float),A);
                //std::cout << "Executing with n: " << n << " incx: "<<incx<<" incy: "<<incy<<" alpha: "<<alpha<<std::endl;
                fb.sger(kernel_names[ix],n,m,alpha,input_x,incx,input_y,incy,input_A,m);
                queue.enqueueReadBuffer(input_A,CL_TRUE,0,n*m*sizeof(float),fpga_res);
                //check
                check_result<float>(n,m,alpha,A,m,x,incx,y,incy,fpga_res);
                test_case++;
            }
        }
    }
}


TEST(TestDGer,TestDger)
{
    const double alphas[nalf] = {0,1.0,0.7};
    cl::CommandQueue queue;
    cl::Context context=fb.get_context();
    cl::Device device=fb.get_device();
    IntelFPGAOCLUtils::createCommandQueue(context,device,queue);
    int incx=1,incy=1;
    double *A,*x,*y,*fpga_res;
    posix_memalign ((void **)&A, IntelFPGAOCLUtils::AOCL_ALIGNMENT, N*M*sizeof(double));
    posix_memalign ((void **)&x, IntelFPGAOCLUtils::AOCL_ALIGNMENT, N*max_inc*sizeof(double));
    posix_memalign ((void **)&y, IntelFPGAOCLUtils::AOCL_ALIGNMENT, M*max_inc*sizeof(double));
    posix_memalign ((void **)&fpga_res, IntelFPGAOCLUtils::AOCL_ALIGNMENT, N*M*sizeof(double));

    cl::Buffer input_A(context, CL_MEM_READ_WRITE|CL_CHANNEL_1_INTELFPGA, N * M*sizeof(double));
    cl::Buffer input_x(context, CL_MEM_READ_ONLY|CL_CHANNEL_2_INTELFPGA, N * max_inc* sizeof(double));
    cl::Buffer input_y(context, CL_MEM_READ_WRITE|CL_CHANNEL_3_INTELFPGA, N * max_inc *sizeof(double));

    std::string kernel_names[]={"test_dger_0","test_dger_1","test_dger_2","test_dger_3"};
    int test_case=1;
    for(int in=0; in<nd;++in)  //Dimension loop
    {
        int n=ndim[in];
        int m=n/2+1;
        //Generate the matrix A
        generate_matrix<double>(A,n,m);


        for(int ix = 0; ix < nincs; ix ++) //incx and incy
        {
            incx=incxs[ix];
            incy=incys[ix];
            int lx = abs(incx) * n;
            int ly = abs(incy) * m;

            //loops over alpha
            for(int ia = 0; ia < nalf ; ia++)
            {
                double alpha=alphas[ia];
                //generate the vectors
                generate_vector<double>(x,lx);
                generate_vector<double>(y,ly);

                //copy everything to device
                queue.enqueueWriteBuffer(input_x,CL_TRUE,0,lx*sizeof(double),x);
                queue.enqueueWriteBuffer(input_y,CL_TRUE,0,ly*sizeof(double),y);
                queue.enqueueWriteBuffer(input_A,CL_TRUE,0,n*m*sizeof(double),A);
                //std::cout << "Executing with n: " << n << " incx: "<<incx<<" incy: "<<incy<<" alpha: "<<alpha<<std::endl;
                fb.dger(kernel_names[ix],n,m,alpha,input_x,incx,input_y,incy,input_A,m);
                queue.enqueueReadBuffer(input_A,CL_TRUE,0,n*m*sizeof(double),fpga_res);
                //check
                check_result<double>(n,m,alpha,A,m,x,incx,y,incy,fpga_res);
                test_case++;
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
void check_result (int n, int m, T alpha, T *A, int lda, T *x, int incx, T *y, int incy, T *fpga_res)
{

    int ix = OFFSET(n, incx);

    for (int i = 0; i < n; i++)
    {
        const T tmp = alpha * x[ix];
        int jy = OFFSET(m, incy);
        for (int j = 0; j < m; j++) {
            A[m* i + j] += y[jy] * tmp;
            jy += incy;
        }
        ix += incx;
    }

CHECK:
    //check result
    for(int i=0;i < n; i++)
    {
        for(int j=0;j<m;j++)
        {
            if (std::is_same<T, float>::value)
                ASSERT_FLOAT_EQ(fpga_res[i*m+j],A[i*m+j]);
            else
                ASSERT_DOUBLE_EQ(fpga_res[i*m+j],A[i*m+j]);
        }
    }
}


