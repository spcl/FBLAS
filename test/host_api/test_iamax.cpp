/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2019 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    Tests for iamax routine.
    Tests ideas borrowed from Blas testing
*/
#include <gtest/gtest.h>
#include <string>
#include <exception>
#include <algorithm>
#include "../../include/utils/ocl_utils.hpp"
#include "../../include/fblas_environment.hpp"
#include "test_tier2.hpp"
FBLASEnvironment fb;

TEST(TestIamax,TestIsamax)
{
    cl::CommandQueue queue;
    cl::Context context=fb.get_context();
    cl::Device device=fb.get_device();
    IntelFPGAOCLUtils::createCommandQueue(context,device,queue);
    float x[N_L1*2];
    int fpga_res;
    cl::Buffer input_x(context, CL_MEM_READ_ONLY|CL_CHANNEL_1_INTELFPGA, N_L1 * 2*sizeof(float));
    cl::Buffer output(context, CL_MEM_WRITE_ONLY|CL_CHANNEL_2_INTELFPGA, sizeof(int));


    std::string kernel_names[]={"test_isamax_0","test_isamax_1"};

    for (int incx = 1; incx <= 1; incx++)
    {
        for (int n = 0; n < N_L1; n++)
        {
            generate_vector<float>(x,n);
            queue.enqueueWriteBuffer(input_x,CL_TRUE,0,std::max(n*abs(incx),1)*sizeof(float),x);
            fb.isamax(kernel_names[incx-1],n,input_x,incx,output);
            queue.enqueueReadBuffer(output,CL_TRUE,0,sizeof(int),&fpga_res);
            //compute the max
            float max=0;
            int result=0;
            int ix = OFFSET(n, incx);
            for (int i = 0; i < n; i++) {
                if (fabs(x[ix]) > max) {
                    max = fabs(x[ix]);
                    result = i;
                }
                ix += incx;
            }
            ASSERT_EQ(fpga_res,result);
        }
    }
}

TEST(TestIamax,TestIdamax)
{
    cl::CommandQueue queue;
    cl::Context context=fb.get_context();
    cl::Device device=fb.get_device();
    IntelFPGAOCLUtils::createCommandQueue(context,device,queue);
    double x[N_L1*2];
    int fpga_res;
    cl::Buffer input_x(context, CL_MEM_READ_ONLY|CL_CHANNEL_1_INTELFPGA, N_L1 * 2*sizeof(double));
    cl::Buffer output(context, CL_MEM_WRITE_ONLY|CL_CHANNEL_2_INTELFPGA, sizeof(int));


    std::string kernel_names[]={"test_idamax_0","test_idamax_1"};

    for (int incx = 1; incx <= 1; incx++)
    {
        for (int n = 0; n < N_L1; n++)
        {
            generate_vector<double>(x,n);
            queue.enqueueWriteBuffer(input_x,CL_TRUE,0,std::max(n*abs(incx),1)*sizeof(double),x);
            fb.idamax(kernel_names[incx-1],n,input_x,incx,output);
            queue.enqueueReadBuffer(output,CL_TRUE,0,sizeof(int),&fpga_res);
            //compute the max
            double max=0;
            int result=0;
            int ix = OFFSET(n, incx);
            for (int i = 0; i < n; i++) {
                if (fabs(x[ix]) > max) {
                    max = fabs(x[ix]);
                    result = i;
                }
                ix += incx;
            }
            ASSERT_EQ(fpga_res,result);
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
