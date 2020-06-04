/**
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

TEST(TestSrotg,TestSrotg)
{

    //actually this not implemented throught an opencl kernel

    std::string kernel_name="test_srotg";
    for(int k=0; k< 8; k++)
    {
        float a=da1[k];
        float b=db1[k];
        float c,s;
        fb.srotg(kernel_name,a,b,c,s);
        ASSERT_FLOAT_EQ(c,dc1[k]);
        ASSERT_FLOAT_EQ(s,ds1[k]);

    }
}


TEST(TestDrotg,TestDrotg)
{

    //actually this not implemented throught an opencl kernel

    std::string kernel_name="test_drotg";
    for(int k=0; k< 8; k++)
    {
        double a=double_da1[k];
        double b=double_db1[k];
        double c,s;
        fb.drotg(kernel_name,a,b,c,s);
        ASSERT_DOUBLE_EQ(c,double_dc1[k]);
        ASSERT_DOUBLE_EQ(s,double_ds1[k]);

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
