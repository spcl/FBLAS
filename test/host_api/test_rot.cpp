/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2019 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    Tests for rot routine.
    Tests ideas borrowed from Blas testing
*/
#include <gtest/gtest.h>
#include <string>
#include <exception>
#include "../../include/utils/ocl_utils.hpp"
#include "../../include/fblas_environment.hpp"
#include "test_tier2.hpp"
FBLASEnvironment fb;


TEST(TestSrot,TestSrot)
{
    cl::CommandQueue queue;
    cl::Context context=fb.get_context();
    cl::Device device=fb.get_device();
    IntelFPGAOCLUtils::createCommandQueue(context,device,queue);
    float fpga_res_x[N_L1],fpga_res_y[N_L1];

    cl::Buffer input_output_x(context, CL_MEM_READ_WRITE|CL_CHANNEL_1_INTELFPGA, N_L1 * sizeof(float));
    cl::Buffer input_output_y(context, CL_MEM_READ_WRITE|CL_CHANNEL_2_INTELFPGA, N_L1 * sizeof(float));
    //copy data only once
    const float sc = .8f;
    const float ss = .6f;
    std::string kernel_names[]={"test_srot_0","test_srot_1","test_srot_2","test_srot_3"};
    for(int ki=0; ki<nincs;ki++)
    {
        int incx=incxs[ki];
        int incy=incys[ki];
        for(int kn=0;kn<4;kn++)
        {
            //copy data
            queue.enqueueWriteBuffer(input_output_x,CL_TRUE,0,N_L1*sizeof(float),dx1);
            queue.enqueueWriteBuffer(input_output_y,CL_TRUE,0,N_L1*sizeof(float),dy1);

            int curr_n=ns[kn];
            fb.srot(kernel_names[ki],curr_n,input_output_x,incx,input_output_y,incy,sc,ss);

            queue.enqueueReadBuffer(input_output_x,CL_TRUE,0,N_L1*sizeof(float),fpga_res_x);
            queue.enqueueReadBuffer(input_output_y,CL_TRUE,0,N_L1*sizeof(float),fpga_res_y);
            //check
            for (int i = 0; i < curr_n; i++) {
                ASSERT_FLOAT_EQ(fpga_res_x[i],dt9x[i+1+ (kn +1+ ((ki+1) << 2)) * 7 - 36]);
                ASSERT_FLOAT_EQ(fpga_res_y[i],dt9y[i+1 + (kn+1 + ((ki+1) << 2)) * 7 - 36]);
            }
        }
    }

}
TEST(TestDrot,TestDrot)
{

    cl::CommandQueue queue;
    cl::Context context=fb.get_context();
    cl::Device device=fb.get_device();
    IntelFPGAOCLUtils::createCommandQueue(context,device,queue);
    double fpga_res_x[N_L1],fpga_res_y[N_L1];

    cl::Buffer input_output_x(context, CL_MEM_READ_WRITE|CL_CHANNEL_1_INTELFPGA, N_L1 * sizeof(double));
    cl::Buffer input_output_y(context, CL_MEM_READ_WRITE|CL_CHANNEL_2_INTELFPGA, N_L1 * sizeof(double));
    //copy data only once
    const double sc = .8;
    const double ss = .6;
    std::string kernel_names[]={"test_drot_0","test_drot_1","test_drot_2","test_drot_3"};
    for(int ki=0; ki<nincs;ki++)
    {
        int incx=incxs[ki];
        int incy=incys[ki];
        for(int kn=0;kn<4;kn++)
        {
            //copy data
            queue.enqueueWriteBuffer(input_output_x,CL_TRUE,0,N_L1*sizeof(double),double_dx1);
            queue.enqueueWriteBuffer(input_output_y,CL_TRUE,0,N_L1*sizeof(double),double_dy1);

            int curr_n=ns[kn];
            fb.drot(kernel_names[ki],curr_n,input_output_x,incx,input_output_y,incy,sc,ss);

            queue.enqueueReadBuffer(input_output_x,CL_TRUE,0,N_L1*sizeof(double),fpga_res_x);
            queue.enqueueReadBuffer(input_output_y,CL_TRUE,0,N_L1*sizeof(double),fpga_res_y);
            //check
            for (int i = 0; i < curr_n; i++) {
                //These are tested as float since it fails for a digit around e-10
                ASSERT_FLOAT_EQ(fpga_res_x[i],double_dt9x[i+1+ (kn +1+ ((ki+1) << 2)) * 7 - 36]);
                ASSERT_FLOAT_EQ(fpga_res_y[i],double_dt9y[i+1 + (kn+1 + ((ki+1) << 2)) * 7 - 36]);

            }
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
