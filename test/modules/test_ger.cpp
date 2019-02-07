/**
  Tests for GEMV routine.
  Tests ideas borrowed from Blas testing
  GEMV check routine is a modified version of the one included in GSL (Gnu Scientific Library) v2.5
*/
#include <gtest/gtest.h>
#include <string>
#include <exception>
#include <algorithm>
#include <string.h>
#include "../../include/utils/ocl_utils.hpp"
#include "../../include/fblas_environment.hpp"
#include "../host_api/test_tier2.hpp"

const int nalf=3;               //number of different alpha values
const int nbeta=3;              // betas
std::string program_path;


template <typename T>
void check_result (int n, int m, T alpha, T *A, int lda, T *x, int incx, T *y, int incy, T *fpga_res);


TEST(TestSGer,TestSger)
{
    const float alphas[nalf] = {0,1.0,0.7};
    cl::Platform platform;
    cl::Device device;
    cl::Context context;
    cl::Program program;
    std::vector<std::string> kernel_names={"test_sger_read_x","test_sger_read_y","test_sger_read_matrix","test_sger","test_sger_write_matrix"};
    std::vector<cl::Kernel> kernels;
    std::vector<cl::CommandQueue> queues;
    IntelFPGAOCLUtils::initEnvironment(platform,device,context,program,program_path,kernel_names,kernels,queues);
    const int num_kernels=kernel_names.size();

    float *A,*x,*y,*fpga_res;
    posix_memalign ((void **)&A, IntelFPGAOCLUtils::AOCL_ALIGNMENT, N*M*sizeof(float));
    posix_memalign ((void **)&x, IntelFPGAOCLUtils::AOCL_ALIGNMENT, N*sizeof(float));
    posix_memalign ((void **)&y, IntelFPGAOCLUtils::AOCL_ALIGNMENT, M*sizeof(float));
    posix_memalign ((void **)&fpga_res, IntelFPGAOCLUtils::AOCL_ALIGNMENT, N*M*sizeof(float));

    cl::Buffer input_A(context, CL_MEM_READ_WRITE|CL_CHANNEL_1_INTELFPGA, N * M*sizeof(float));
    cl::Buffer input_x(context, CL_MEM_READ_ONLY|CL_CHANNEL_2_INTELFPGA, N * sizeof(float));
    cl::Buffer input_y(context, CL_MEM_READ_WRITE|CL_CHANNEL_3_INTELFPGA, N * sizeof(float));

    int tile_size=128;
    for(int in=0; in<nd;++in)  //Dimension loop
    {
        int n=ndim[in];
        int m=n/2+1;
        //Generate the matrix A
        generate_matrix<float>(A,n,m);
        queues[0].enqueueWriteBuffer(input_A,CL_TRUE,0,n*m*sizeof(float),A);
        //non transposed version

        //loops over alpha and beta
        for(int ia = 0; ia < nalf ; ia++)
        {
            float alpha=alphas[ia];
            //generate the vectors
            generate_vector<float>(x,n);
            generate_vector<float>(y,m);

            queues[0].enqueueWriteBuffer(input_x,CL_TRUE,0,n*sizeof(float),x);
            queues[0].enqueueWriteBuffer(input_y,CL_TRUE,0,m*sizeof(float),y);
            int x_repetitions=1;
            int y_repetitions=ceil((float)(n)/tile_size);
            kernels[0].setArg(0,sizeof(cl_mem),&input_x);
            kernels[0].setArg(1,sizeof(int),&n);
            kernels[0].setArg(2,sizeof(int),&tile_size);
            kernels[0].setArg(3,sizeof(int),&x_repetitions);
            kernels[1].setArg(0,sizeof(cl_mem),&input_y);
            kernels[1].setArg(1,sizeof(int),&m);
            kernels[1].setArg(2,sizeof(int),&tile_size);
            kernels[1].setArg(3,sizeof(int),&y_repetitions);
            //read matrix
            kernels[2].setArg(0,sizeof(cl_mem),&input_A);
            kernels[2].setArg(1,sizeof(int),&n);
            kernels[2].setArg(2,sizeof(int),&m);
            kernels[2].setArg(3,sizeof(int),&m);

            //gemv
            kernels[3].setArg(0,sizeof(float),&alpha);
            kernels[3].setArg(1,sizeof(int),&n);
            kernels[3].setArg(2,sizeof(int),&m);


            //write matrix
            kernels[4].setArg(0,sizeof(cl_mem),&input_A);
            kernels[4].setArg(1,sizeof(int),&n);
            kernels[4].setArg(2,sizeof(int),&m);
            kernels[4].setArg(3,sizeof(int),&m);
            for(int i=0;i<num_kernels;i++)
               queues[i].enqueueTask(kernels[i]);

            //wait
            for(int i=0;i<num_kernels;i++)
               queues[i].finish();

            queues[0].enqueueReadBuffer(input_A,CL_TRUE,0,n*m*sizeof(float),fpga_res);
            //check
            check_result<float>(n,m,alpha,A,m,x,1,y,1,fpga_res);

        }

    }
    //buffers and other objects are automatically deleted at program exit
}

int main(int argc, char *argv[])
{
    if(argc<2)
    {
        std::cerr << "Usage: env CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1 " << argv[0] << " <fpga binary file>" << std::endl;
        return -1;
    }
    program_path=argv[1];
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

