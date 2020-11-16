/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2020 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    Tests for swap routine.
    Tests ideas borrowed from Blas testing
*/
#include <gtest/gtest.h>
#include <string>
#include <exception>
#include "../../include/utils/ocl_utils.hpp"
#include "../../include/fblas_environment.hpp"
#include "test_tier2.hpp"
FBLASEnvironment fb;

TEST(TestSswap,TestSswap)
{
    cl::CommandQueue queue;
    cl::Context context=fb.get_context();
    cl::Device device=fb.get_device();
    IntelFPGAOCLUtils::createCommandQueue(context,device,queue);
    float fpga_res_x[N];
    float fpga_res_y[N];

    cl::Buffer input_output_x(context, CL_MEM_READ_WRITE, N_L1 * sizeof(float));
    cl::Buffer input_output_y(context, CL_MEM_READ_WRITE, N_L1 * sizeof(float));

    std::string kernel_names[]={"test_sswap_0","test_sswap_1","test_sswap_2","test_sswap_3"};
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
            fb.sswap(kernel_names[ki],curr_n,input_output_x,incx,input_output_y,incy);
            queue.enqueueReadBuffer(input_output_x,CL_TRUE,0,N_L1*sizeof(float),fpga_res_x);
            queue.enqueueReadBuffer(input_output_y,CL_TRUE,0,N_L1*sizeof(float),fpga_res_y);
            //check

            for (int i = 0; i < curr_n; i++) {
                ASSERT_FLOAT_EQ(fpga_res_x[i],dt10x[i+1 + (kn+1 + ((ki+1) << 2)) * 7 - 36]);
                ASSERT_FLOAT_EQ(fpga_res_y[i],dt10y[i+1 + (kn+1 + ((ki+1) << 2)) * 7 - 36]);
            }
        }
    }

}

TEST(TestDswap,TestDswap)
{
    cl::CommandQueue queue;
    cl::Context context=fb.get_context();
    cl::Device device=fb.get_device();
    IntelFPGAOCLUtils::createCommandQueue(context,device,queue);
    double fpga_res_x[N];
    double fpga_res_y[N];

    cl::Buffer input_output_x(context, CL_MEM_READ_WRITE, N_L1 * sizeof(double));
    cl::Buffer input_output_y(context, CL_MEM_READ_WRITE, N_L1 * sizeof(double));

    std::string kernel_names[]={"test_dswap_0","test_dswap_1","test_dswap_2","test_dswap_3"};
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
            fb.dswap(kernel_names[ki],curr_n,input_output_x,incx,input_output_y,incy);
            queue.enqueueReadBuffer(input_output_x,CL_TRUE,0,N_L1*sizeof(double),fpga_res_x);
            queue.enqueueReadBuffer(input_output_y,CL_TRUE,0,N_L1*sizeof(double),fpga_res_y);
            //check

            for (int i = 0; i < curr_n; i++) {
                ASSERT_DOUBLE_EQ(fpga_res_x[i],double_dt10x[i+1 + (kn+1 + ((ki+1) << 2)) * 7 - 36]);
                ASSERT_DOUBLE_EQ(fpga_res_y[i],double_dt10y[i+1 + (kn+1 + ((ki+1) << 2)) * 7 - 36]);
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
