/**
  Tests for dot routine.
  Tests ideas borrowed from Blas testing
*/
#include <gtest/gtest.h>
#include <string>
#include <exception>
#include "../../include/utils/ocl_utils.hpp"

static int ns[4] = { 0,1,2,4 };
static const int N=7;
static int incxs[4] = { 1,2,-2,-1 };
static int incys[4] = { 1,-2,1,-2 };
std::string program_path;
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
    float sa=0.3f;
    cl::Platform platform;
    cl::Device device;
    cl::Context context;
    cl::Program program;
    std::vector<std::string> kernel_names={"test_axpy_read_x","test_axpy_read_y","test_axpy","test_axpy_write_vector"};
    std::vector<cl::Kernel> kernels;
    std::vector<cl::CommandQueue> queues;
    IntelFPGAOCLUtils::initEnvironment(platform,device,context,program,program_path,kernel_names,kernels,queues);
    const int num_kernels=kernel_names.size();

    float fpga_res[N];
    cl::Buffer input_x(context, CL_MEM_READ_ONLY|CL_CHANNEL_1_INTELFPGA, N * sizeof(float));
    cl::Buffer input_y(context, CL_MEM_READ_WRITE|CL_CHANNEL_2_INTELFPGA, N * sizeof(float));
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

        kernels[2].setArg(0,sizeof(float),&sa);
        kernels[2].setArg(1,sizeof(int),&curr_n);

        kernels[3].setArg(0,sizeof(cl_mem),&input_y);
        kernels[3].setArg(1,sizeof(int),&curr_n);
        kernels[3].setArg(2,sizeof(int),&width);
        for(int i=0;i<num_kernels;i++)
           queues[i].enqueueTask(kernels[i]);

        //wait
        for(int i=0;i<num_kernels;i++)
           queues[i].finish();

        queues[0].enqueueReadBuffer(input_y,CL_TRUE,0,N*sizeof(float),&fpga_res);
        //check
        for (int j = 0; j < curr_n; j++) {
            ASSERT_FLOAT_EQ(fpga_res[j],dt8[j+1 + (kn+1+ ((1) << 2)) * 7 - 36]);
        }
        //reset y
        queues[0].enqueueWriteBuffer(input_y,CL_TRUE,0,N*sizeof(float),dy1);
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
