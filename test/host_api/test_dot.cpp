/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2019 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    Tests for dot routine.
    Tests ideas borrowed from Blas testing
*/
#include <gtest/gtest.h>
#include <string>
#include <exception>
#include "../../include/utils/ocl_utils.hpp"
#include "../../include/fblas_environment.hpp"

FBLASEnvironment fb;
static int ns[4] = { 0,1,2,4 };
static const int N=7;
static int incxs[4] = { 1,2,-2,-1 };
static int incys[4] = { 1,-2,1,-2 };

//g++ test_dot.cpp -lgtest $( aocl compile-config ) -std=c++11 $(aocl link-config) -lpthread -I/home/tdematt/lib/rapidjson/include
TEST(TestDot,TestSdot)
{
    static float dx1[7] = { .6f,.1f,-.5f,.8f,.9f,-.3f,-.4f };
    static float dy1[7] = { .5f,-.9f,.3f,.7f,-.6f,.2f,.8f };
    static float dt7[16] = { 0.f,.3f,.21f,.62f,0.f,.3f,-.07f,
                .85f,0.f,.3f,-.79f,-.74f,0.f,.3f,.33f,1.27f };
    cl::CommandQueue queue;
    cl::Context context=fb.get_context();
    cl::Device device=fb.get_device();
    IntelFPGAOCLUtils::createCommandQueue(context,device,queue);
    int incx=1,incy=1;
    float fpga_res;
    cl::Buffer output(context,CL_MEM_WRITE_ONLY,sizeof(float));

    cl::Buffer input_x(context, CL_MEM_READ_ONLY|CL_CHANNEL_1_INTELFPGA, N * sizeof(float));
    cl::Buffer input_y(context, CL_MEM_READ_ONLY|CL_CHANNEL_2_INTELFPGA, N * sizeof(float));
    //copy data only once
    queue.enqueueWriteBuffer(input_x,CL_TRUE,0,N*sizeof(float),dx1);
    queue.enqueueWriteBuffer(input_y,CL_TRUE,0,N*sizeof(float),dy1);

    std::string kernel_names[]={"test_sdot_0","test_sdot_1","test_sdot_2","test_sdot_3"};
    for(int ki=0; ki<4;ki++) 
    {
        int incx=incxs[ki];
        int incy=incys[ki];
        for(int kn=0;kn<4;kn++)
        {

            int curr_n=ns[kn];
            fb.sdot(kernel_names[ki],(unsigned int)curr_n,input_x,incx, input_y, incy,output);
            queue.enqueueReadBuffer(output,CL_TRUE,0,sizeof(float),&fpga_res);
            //check
            ASSERT_FLOAT_EQ(fpga_res,dt7[ki*4+kn]);
        }
    }

    //buffers and other objects are automatically deleted at program exit

}

TEST(TestDot, TestDdot)
{
    const double dx1[7] = { .6,.1,-.5,.8,.9,-.3,-.4 };
    const double dy1[7] = { .5,-.9,.3,.7,-.6,.2,.8 };
    const double dt7[16]= { 0.,.3,.21,.62,0.,.3,-.07,
        .85,0.,.3,-.79,-.74,0.,.3,.33,1.27 };

    cl::CommandQueue queue;
    cl::Context context=fb.get_context();
    cl::Device device=fb.get_device();
    IntelFPGAOCLUtils::createCommandQueue(context,device,queue);
    int incx=1,incy=1;
    double fpga_res;
    cl::Buffer output(context,CL_MEM_WRITE_ONLY,sizeof(double));

    cl::Buffer input_x(context, CL_MEM_READ_ONLY|CL_CHANNEL_1_INTELFPGA, N * sizeof(double));
    cl::Buffer input_y(context, CL_MEM_READ_ONLY|CL_CHANNEL_2_INTELFPGA, N * sizeof(double));
    //copy data only once
    queue.enqueueWriteBuffer(input_x,CL_TRUE,0,N*sizeof(double),dx1);
    queue.enqueueWriteBuffer(input_y,CL_TRUE,0,N*sizeof(double),dy1);

    std::string kernel_names[]={"test_ddot_0","test_ddot_1","test_ddot_2","test_ddot_3"};
    for(int ki=0; ki<4;ki++) 
    {
        int incx=incxs[ki];
        int incy=incys[ki];
        for(int kn=0;kn<4;kn++)
        {

            int curr_n=ns[kn];
            fb.ddot(kernel_names[ki],(unsigned int)curr_n,input_x,incx, input_y, incy,output);
            queue.enqueueReadBuffer(output,CL_TRUE,0,sizeof(double),&fpga_res);
            //check
            ASSERT_DOUBLE_EQ(fpga_res,dt7[ki*4+kn]);
        }
    }

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
