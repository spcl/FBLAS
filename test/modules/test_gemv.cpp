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

//g++ test_dot.cpp -lgtest $( aocl compile-config ) -std=c++11 $(aocl link-config) -lpthread -I/home/tdematt/lib/rapidjson/include

template <typename T>
void check_result ( int n, int m, T alpha, T *A, int lda, T *x, int incx, T beta, T *y, int incy, T *fpga_res)
{
    int len_x=m, len_y=n;
    int i,j;
    if (m != 0 && n != 0)
    {

        if (alpha != 0.0f || beta != 1.0)
        {

            /* form  y := beta*y */
            if (beta == 0.0) {
                int iy = OFFSET(len_y, incy);
                for (i = 0; i < len_y; i++) {
                    y[iy] = 0.0;
                    iy += incy;
                }
            }
            else if (beta != 1.0f)
            {
                int iy = OFFSET(len_y, incy);
                for (i = 0; i < len_y; i++) {
                    y[iy] *= beta;
                    iy += incy;
                }
            }

            if (alpha != 0.0)
            {

                /* form  y := alpha*A*x + y */
                int iy = OFFSET(len_y, incy);
                for (i = 0; i < len_y; i++) {
                    T temp = 0.0;
                    int ix = OFFSET(len_x, incx);
                    for (j = 0; j < len_x; j++) {
                        temp += x[ix] * A[lda * i + j];
                        ix += incx;
                    }
                    y[iy] += alpha * temp;
                    iy += incy;
                }

            }
        }
    }

CHECK:
    //check result using nrm1
    int iy = OFFSET(len_y, incy);
    //Measure error by considering
    T nrm1_diff=0, nrm1_orig=0;
    T error;
    for(int i=0;i < len_y; i++)
    {
        nrm1_diff+=abs(fpga_res[iy]-y[iy]);
        nrm1_orig+=abs(y[iy]);
        iy+=incy;
    }
    if(nrm1_diff ==0 && nrm1_orig ==0)
        error=0;
    else
        error=nrm1_diff/nrm1_orig;
    if (std::is_same<T, float>::value)
        ASSERT_LE(error,flteps);
    else
        ASSERT_LE(error,dbleps);
}


TEST(TestSGemv,TestSgemv)
{
    const float alphas[nalf] = {0,1.0,0.7};
    const float betas[nbeta] = {0,1.0,0.9};
    cl::Platform platform;
    cl::Device device;
    cl::Context context;
    cl::Program program;
    std::vector<std::string> kernel_names={"test_sgemv_read_x","test_sgemv_read_y","test_sgemv_read_matrix","test_sgemv","test_sgemv_write_vector"};
    std::vector<cl::Kernel> kernels;
    std::vector<cl::CommandQueue> queues;
    IntelFPGAOCLUtils::initEnvironment(platform,device,context,program,program_path,kernel_names,kernels,queues);
    const int num_kernels=kernel_names.size();

    float *A,*x,*y,*fpga_res;
    posix_memalign ((void **)&A, IntelFPGAOCLUtils::AOCL_ALIGNMENT, N*M*sizeof(float));
    posix_memalign ((void **)&x, IntelFPGAOCLUtils::AOCL_ALIGNMENT, N*sizeof(float));
    posix_memalign ((void **)&y, IntelFPGAOCLUtils::AOCL_ALIGNMENT, M*sizeof(float));
    posix_memalign ((void **)&fpga_res, IntelFPGAOCLUtils::AOCL_ALIGNMENT, N*sizeof(float));

    cl::Buffer input_A(context, CL_MEM_READ_ONLY|CL_CHANNEL_1_INTELFPGA, N * M*sizeof(float));
    cl::Buffer input_x(context, CL_MEM_READ_ONLY|CL_CHANNEL_2_INTELFPGA, N * sizeof(float));
    cl::Buffer input_y(context, CL_MEM_READ_WRITE|CL_CHANNEL_3_INTELFPGA, N * sizeof(float));

    int width=16;
    int tile_size=128;
    int one=1;
    for(int in=0; in<nd;++in)  //Dimension loop
    {
        int n=ndim[in];
        int m=n/2+1;
        //Generate the matrix A
        generate_matrix<float>(A,n,m);
        queues[0].enqueueWriteBuffer(input_A,CL_TRUE,0,n*m*sizeof(float),A);
        //non transposed version
        int nl=n, ml=m;

        int lx = ml;
        int ly = nl;

        //loops over alpha and beta
        for(int ia = 0; ia < nalf ; ia++)
        {
            float alpha=alphas[ia];
            for(int ib = 0; ib < nbeta; ib++ )
            {
                float beta = betas[ib];
                //generate the vectors
                generate_vector<float>(x,lx);
                generate_vector<float>(y,ly);

                queues[0].enqueueWriteBuffer(input_x,CL_TRUE,0,lx*sizeof(float),x);
                queues[0].enqueueWriteBuffer(input_y,CL_TRUE,0,ly*sizeof(float),y);
                int x_repetitions=ceil((float)(ly)/tile_size);
                int y_repetitions=(beta==0)?0:1;
                kernels[0].setArg(0,sizeof(cl_mem),&input_x);
                kernels[0].setArg(1,sizeof(int),&lx);
                kernels[0].setArg(2,sizeof(int),&tile_size);
                kernels[0].setArg(3,sizeof(int),&x_repetitions);
                kernels[1].setArg(0,sizeof(cl_mem),&input_y);
                kernels[1].setArg(1,sizeof(int),&ly);
                kernels[1].setArg(2,sizeof(int),&tile_size);
                kernels[1].setArg(3,sizeof(int),&y_repetitions);
                //read matrix
                kernels[2].setArg(0,sizeof(cl_mem),&input_A);
                kernels[2].setArg(1,sizeof(int),&ly);
                kernels[2].setArg(2,sizeof(int),&lx);
                kernels[2].setArg(3,sizeof(int),&lx);

                //gemv
                kernels[3].setArg(0,sizeof(int),&one);
                kernels[3].setArg(1,sizeof(int),&ly);
                kernels[3].setArg(2,sizeof(int),&lx);
                kernels[3].setArg(3,sizeof(float),&alpha);
                kernels[3].setArg(4,sizeof(float),&beta);

                //write
                kernels[4].setArg(0,sizeof(cl_mem),&input_y);
                kernels[4].setArg(1,sizeof(int),&ly);
                kernels[4].setArg(2,sizeof(int),&tile_size);
                for(int i=0;i<num_kernels;i++)
                   queues[i].enqueueTask(kernels[i]);

                //wait
                for(int i=0;i<num_kernels;i++)
                   queues[i].finish();




                queues[0].enqueueReadBuffer(input_y,CL_TRUE,0,ly*sizeof(float),fpga_res);
                //check
                check_result<float>(n,m,alpha,A,m,x,1,beta,y,1,fpga_res);
            }
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
