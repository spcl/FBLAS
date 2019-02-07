/**

    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2019 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    Testing

    Tests for axpy routine.
    Tests ideas borrowed from Blas testing
*/
#include <gtest/gtest.h>
#include <string>
#include <exception>
#include "../../include/utils/ocl_utils.hpp"
#include "../../include/fblas_environment.hpp"

FBLASEnvironment fb;
const int ns[4] = { 0,1,2,4 };
const int N=7;
const int incxs[4] = { 1,2,-2,-1 };
const int incys[4] = { 1,-2,1,-2 };

//g++ test_dot.cpp -lgtest $( aocl compile-config ) -std=c++11 $(aocl link-config) -lpthread -I/home/tdematt/lib/rapidjson/include
TEST(TestAxpy,TestSaxpy)
{

    const float dx1[7] = { .6f,.1f,-.5f,.8f,.9f,-.3f,-.4f };
    const float dy1[7] = { .5f,-.9f,.3f,.7f,-.6f,.2f,.8f };
    const float dt8[112] = { .5f,0.f,0.f,0.f,0.f,
        0.f,0.f,.68f,0.f,0.f,0.f,0.f,0.f,0.f,.68f,-.87f,0.f,0.f,0.f,0.f,
        0.f,.68f,-.87f,.15f,.94f,0.f,0.f,0.f,.5f,0.f,0.f,0.f,0.f,0.f,0.f,
        .68f,0.f,0.f,0.f,0.f,0.f,0.f,.35f,-.9f,.48f,0.f,0.f,0.f,0.f,.38f,
        -.9f,.57f,.7f,-.75f,.2f,.98f,.5f,0.f,0.f,0.f,0.f,0.f,0.f,.68f,0.f,
        0.f,0.f,0.f,0.f,0.f,.35f,-.72f,0.f,0.f,0.f,0.f,0.f,.38f,-.63f,
        .15f,.88f,0.f,0.f,0.f,.5f,0.f,0.f,0.f,0.f,0.f,0.f,.68f,0.f,0.f,
        0.f,0.f,0.f,0.f,.68f,-.9f,.33f,0.f,0.f,0.f,0.f,.68f,-.9f,.33f,.7f,
        -.75f,.2f,1.04f };
    const float sa=0.3f;
    cl::CommandQueue queue;
    cl::Context context=fb.get_context();
    cl::Device device=fb.get_device();
    IntelFPGAOCLUtils::createCommandQueue(context,device,queue);
    int incx=1,incy=1;
    float fpga_res[N];
    cl::Buffer output(context,CL_MEM_WRITE_ONLY,sizeof(float));

    cl::Buffer input_x(context, CL_MEM_READ_ONLY|CL_CHANNEL_1_INTELFPGA, N * sizeof(float));
    cl::Buffer input_y(context, CL_MEM_READ_WRITE|CL_CHANNEL_2_INTELFPGA, N * sizeof(float));
    //copy data only once
    queue.enqueueWriteBuffer(input_x,CL_TRUE,0,N*sizeof(float),dx1);
    queue.enqueueWriteBuffer(input_y,CL_TRUE,0,N*sizeof(float),dy1);

    std::string kernel_names[]={"test_saxpy_0","test_saxpy_1","test_saxpy_2","test_saxpy_3"};
    for(int ki=0; ki<4;++ki) 
    {
        int incx=incxs[ki];
        int incy=incys[ki];
        for(int kn=0;kn<4;++kn)
        {

            int curr_n=ns[kn];
            fb.saxpy(kernel_names[ki],(unsigned int)curr_n,sa,input_x,incx, input_y, incy);
            queue.enqueueReadBuffer(input_y,CL_TRUE,0,N*sizeof(float),&fpga_res);

            //check
            for (int j = 0; j < curr_n; j++) {
                ASSERT_FLOAT_EQ(fpga_res[j],dt8[j+1 + (kn+1+ ((ki+1) << 2)) * 7 - 36]);
            }
            //reset y
            queue.enqueueWriteBuffer(input_y,CL_TRUE,0,N*sizeof(float),dy1);
        }
    }
    //buffers and other objects are automatically deleted at program exit
}

TEST(TestAxpy,TestDaxpy)
{

    const double dx1[7] = { .6,.1,-.5,.8,.9,-.3,-.4 };
    const double dy1[7] = { .5,-.9,.3,.7,-.6,.2,.8 };
    const double dt8[112] = { .5,0.,0.,0.,0.,0.,0.,
        .68,0.,0.,0.,0.,0.,0.,.68,-.87,0.,0.,0.,0.,0.,.68,-.87,.15,.94,0.,
        0.,0.,.5,0.,0.,0.,0.,0.,0.,.68,0.,0.,0.,0.,0.,0.,.35,-.9,.48,0.,
        0.,0.,0.,.38,-.9,.57,.7,-.75,.2,.98,.5,0.,0.,0.,0.,0.,0.,.68,0.,
        0.,0.,0.,0.,0.,.35,-.72,0.,0.,0.,0.,0.,.38,-.63,.15,.88,0.,0.,0.,
        .5,0.,0.,0.,0.,0.,0.,.68,0.,0.,0.,0.,0.,0.,.68,-.9,.33,0.,0.,0.,
        0.,.68,-.9,.33,.7,-.75,.2,1.04 };
    const double sa=0.3;
    cl::CommandQueue queue;
    cl::Context context=fb.get_context();
    cl::Device device=fb.get_device();
    IntelFPGAOCLUtils::createCommandQueue(context,device,queue);
    int incx,incy;
    double fpga_res[N];
    cl::Buffer output(context,CL_MEM_WRITE_ONLY,sizeof(double));

    cl::Buffer input_x(context, CL_MEM_READ_ONLY|CL_CHANNEL_1_INTELFPGA, N * sizeof(double));
    cl::Buffer input_y(context, CL_MEM_READ_WRITE|CL_CHANNEL_2_INTELFPGA, N * sizeof(double));
    //copy data only once
    queue.enqueueWriteBuffer(input_x,CL_TRUE,0,N*sizeof(double),dx1);
    queue.enqueueWriteBuffer(input_y,CL_TRUE,0,N*sizeof(double),dy1);

    std::string kernel_names[]={"test_daxpy_0","test_daxpy_1","test_daxpy_2","test_daxpy_3"};
    for(int ki=0; ki<4;++ki) 
    {
        int incx=incxs[ki];
        int incy=incys[ki];
        for(int kn=0;kn<4;++kn)
        {

            int curr_n=ns[kn];
            fb.daxpy(kernel_names[ki],(unsigned int)curr_n,sa,input_x,incx, input_y, incy);
            queue.enqueueReadBuffer(input_y,CL_TRUE,0,N*sizeof(double),&fpga_res);

            //check
            for (int j = 0; j < curr_n; j++) {
                ASSERT_DOUBLE_EQ(fpga_res[j],dt8[j+1 + (kn+1+ ((ki+1) << 2)) * 7 - 36]);
            }
            //reset y
            queue.enqueueWriteBuffer(input_y,CL_TRUE,0,N*sizeof(double),dy1);
        }
    }
    //buffers and other objects are automatically deleted at program exit
}



int main(int argc, char *argv[])
{
    if(argc<3)
    {
        std::cerr << "Usage: [env CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1 " << argv[0] << " <fpga binary file> <json description>" << std::endl;
        return -1;
    }
    std::string program_path=argv[1];
    std::string json_path=argv[2];
    fb =FBLASEnvironment (program_path,json_path);
        ::testing::InitGoogleTest(&argc, argv);
         return RUN_ALL_TESTS();
}
