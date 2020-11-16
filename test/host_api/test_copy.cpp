/**

    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2020 ETH-Zurich. All rights reserved.
    See LICENSE for license information.


  Tests for copy routine.
  Tests ideas borrowed from Blas testing
*/
#include <gtest/gtest.h>
#include <string>
#include <exception>
#include "../../include/utils/ocl_utils.hpp"
#include "../../include/fblas_environment.hpp"
#include "test_tier2.hpp"
FBLASEnvironment fb;

TEST(TestScopy,TestScopy)
{

    cl::CommandQueue queue;
    cl::Context context=fb.get_context();
    cl::Device device=fb.get_device();
    IntelFPGAOCLUtils::createCommandQueue(context,device,queue);
    float fpga_res[N_L1];

    cl::Buffer input_x(context, CL_MEM_READ_ONLY|CL_CHANNEL_1_INTELFPGA, N_L1 * sizeof(float));
    cl::Buffer output_y(context, CL_MEM_READ_WRITE|CL_CHANNEL_2_INTELFPGA, N_L1 * sizeof(float));
    //copy data only once
    queue.enqueueWriteBuffer(input_x,CL_TRUE,0,N_L1*sizeof(float),dx1);

    std::string kernel_names[]={"test_scopy_0","test_scopy_1","test_scopy_2","test_scopy_3"};
    for(int ki=0; ki<nincs;ki++)
    {
        int incx=incxs[ki];
        int incy=incys[ki];
        for(int kn=0;kn<4;kn++)
        {
            //recopy y
            queue.enqueueWriteBuffer(output_y,CL_TRUE,0,N_L1*sizeof(float),dy1);

            int curr_n=ns[kn];
            fb.scopy(kernel_names[ki],curr_n,input_x,incx,output_y,incy);
            queue.enqueueReadBuffer(output_y,CL_TRUE,0,N_L1*sizeof(float),fpga_res);
            //check
            for (int i = 0; i < curr_n; i++) {
                ASSERT_FLOAT_EQ(fpga_res[i],dt10y[i+1 + (kn+1 + ((ki+1) << 2)) * 7 - 36]);
            }
        }
    }

}


TEST(TestDcopy,TestDcopy)
{

    cl::CommandQueue queue;
    cl::Context context=fb.get_context();
    cl::Device device=fb.get_device();
    IntelFPGAOCLUtils::createCommandQueue(context,device,queue);
    double fpga_res[N_L1];

    cl::Buffer input_x(context, CL_MEM_READ_ONLY|CL_CHANNEL_1_INTELFPGA, N_L1 * sizeof(double));
    cl::Buffer output_y(context, CL_MEM_READ_WRITE|CL_CHANNEL_2_INTELFPGA, N_L1 * sizeof(double));
    //copy data only once
    queue.enqueueWriteBuffer(input_x,CL_TRUE,0,N_L1*sizeof(double),double_dx1);

    std::string kernel_names[]={"test_dcopy_0","test_dcopy_1","test_dcopy_2","test_dcopy_3"};
    for(int ki=0; ki<nincs;ki++)
    {
        int incx=incxs[ki];
        int incy=incys[ki];
        for(int kn=0;kn<4;kn++)
        {
            //recopy y
            queue.enqueueWriteBuffer(output_y,CL_TRUE,0,N_L1*sizeof(double),double_dy1);

            int curr_n=ns[kn];
            fb.dcopy(kernel_names[ki],curr_n,input_x,incx,output_y,incy);
            queue.enqueueReadBuffer(output_y,CL_TRUE,0,N_L1*sizeof(double),fpga_res);
            //check
            for (int i = 0; i < curr_n; i++) {
                ASSERT_DOUBLE_EQ(fpga_res[i],double_dt10y[i+1 + (kn+1 + ((ki+1) << 2)) * 7 - 36]);
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
