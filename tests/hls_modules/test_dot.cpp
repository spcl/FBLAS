/**
  Tests for dot routine.
  Tests ideas borrowed from Blas testing
*/
#include <gtest/gtest.h>
#include <string>
#include <exception>
#include "../../include/utils/ocl_utils.hpp"

#if !defined(CL_CHANNEL_1_INTELFPGA)
// include this header if channel macros are not defined in cl.hpp (versions >=19.0)
#include "CL/cl_ext_intelfpga.h"
#endif

static int ns[4] = { 0,1,2,4 };
static const int N=7;
static int incxs[4] = { 1,2,-2,-1 };
static int incys[4] = { 1,-2,1,-2 };
std::string program_path;
//g++ test_dot.cpp -lgtest $( aocl compile-config ) -std=c++11 $(aocl link-config) -lpthread -I/home/tdematt/lib/rapidjson/include
TEST(TestDot,TestSdot)
{
    static float dx1[7] = { .6f,.1f,-.5f,.8f,.9f,-.3f,-.4f };
    static float dy1[7] = { .5f,-.9f,.3f,.7f,-.6f,.2f,.8f };
    static float dt7[16] = { 0.f,.3f,.21f,.62f,0.f,.3f,-.07f,
                .85f,0.f,.3f,-.79f,-.74f,0.f,.3f,.33f,1.27f };
    cl::Platform platform;
    cl::Device device;
    cl::Context context;
    cl::Program program;
    std::vector<std::string> kernel_names={"test_sdot_read_x","test_sdot_read_y","test_sdot","test_sdot_sink"};
    std::vector<cl::Kernel> kernels;
    std::vector<cl::CommandQueue> queues;
    IntelFPGAOCLUtils::initEnvironment(platform,device,context,program,program_path,kernel_names,kernels,queues);
    const int num_kernels=kernel_names.size();

    int incx=1,incy=1;
    float fpga_res;
    cl::Buffer output(context,CL_MEM_WRITE_ONLY,sizeof(float));

    cl::Buffer input_x(context, CL_MEM_READ_ONLY|CL_CHANNEL_1_INTELFPGA, N * sizeof(float));
    cl::Buffer input_y(context, CL_MEM_READ_ONLY|CL_CHANNEL_2_INTELFPGA, N * sizeof(float));
    //copy data only once
    queues[0].enqueueWriteBuffer(input_x,CL_TRUE,0,N*sizeof(float),dx1);
    queues[0].enqueueWriteBuffer(input_y,CL_TRUE,0,N*sizeof(float),dy1);
    int width=16;
    int one=1;

    for(int kn=0;kn<4;kn++)
    {

        int curr_n=ns[kn];
        kernels[0].setArg(0,sizeof(cl_mem),&input_x);
        kernels[0].setArg(1,sizeof(int),&curr_n);
        kernels[0].setArg(2,sizeof(int),&width);
        kernels[0].setArg(3,sizeof(int),&one);
        kernels[1].setArg(0,sizeof(cl_mem),&input_y);
        kernels[1].setArg(1,sizeof(int),&curr_n);
        kernels[1].setArg(2,sizeof(int),&width);
        kernels[1].setArg(3,sizeof(int),&one);
        kernels[2].setArg(0,sizeof(int),&curr_n);
        kernels[3].setArg(0,sizeof(cl_mem),&output);
        for(int i=0;i<num_kernels;i++)
           queues[i].enqueueTask(kernels[i]);

        //wait
        for(int i=0;i<num_kernels;i++)
           queues[i].finish();

        queues[0].enqueueReadBuffer(output,CL_TRUE,0,sizeof(float),&fpga_res);
        //check
        ASSERT_FLOAT_EQ(fpga_res,dt7[kn]);
   }


    //buffers and other objects are automatically deleted at program exit

}



int main(int argc, char *argv[])
{
    if(argc<2)
    {
        std::cerr << "Usage: [env CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1 " << argv[0] << " <fpga binary file> " << std::endl;
        return -1;
    }
    program_path=argv[1];
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
