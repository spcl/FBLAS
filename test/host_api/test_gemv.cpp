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
#include "test_tier2.hpp"
#define GTEST_COUT std::cerr << "[          ] [ INFO ]"

FBLASEnvironment fb;
const int nalf=3;               //number of different alpha values
const int nbeta=3;              // betas

//g++ test_dot.cpp -lgtest $( aocl compile-config ) -std=c++11 $(aocl link-config) -lpthread -I/home/tdematt/lib/rapidjson/include

template <typename T>
void check_result (bool trans, int n, int m, T alpha, T *A, int lda, T *x, int incx, T beta, T *y, int incy, T *fpga_res)
{
    //The implementation of GEMV has been
    int len_x, len_y;
    int i,j;
    if (!trans) {
        len_x = m;
        len_y = n;
    } else {
        len_x = n;
        len_y = m;
    }
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

                if (!trans) {
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
                } else {
                    /* form  y := alpha*A'*x + y */
                    int ix = OFFSET(len_x, incx);
                    for (j = 0; j < len_x; j++) {
                        const T temp = alpha * x[ix];
                        if (temp != 0.0) {
                            int iy = OFFSET(len_y, incy);
                            for (i = 0; i < len_y; i++) {
                                y[iy] += temp * A[lda * j + i];
                                iy += incy;
                            }
                        }
                        ix += incx;
                    }

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
    cl::CommandQueue queue;
    cl::Context context=fb.get_context();
    cl::Device device=fb.get_device();
    IntelFPGAOCLUtils::createCommandQueue(context,device,queue);
    int incx=1,incy=1;
    float *A,*x,*y,*fpga_res;
    posix_memalign ((void **)&A, IntelFPGAOCLUtils::AOCL_ALIGNMENT, N*M*sizeof(float));
    posix_memalign ((void **)&x, IntelFPGAOCLUtils::AOCL_ALIGNMENT, N*max_inc*sizeof(float));
    posix_memalign ((void **)&y, IntelFPGAOCLUtils::AOCL_ALIGNMENT, M*max_inc*sizeof(float));
    posix_memalign ((void **)&fpga_res, IntelFPGAOCLUtils::AOCL_ALIGNMENT, N*max_inc*sizeof(float));
    
    cl::Buffer input_A(context, CL_MEM_READ_ONLY|CL_CHANNEL_1_INTELFPGA, N * M*sizeof(float));
    cl::Buffer input_x(context, CL_MEM_READ_ONLY|CL_CHANNEL_2_INTELFPGA, N * max_inc* sizeof(float));
    cl::Buffer input_y(context, CL_MEM_READ_WRITE|CL_CHANNEL_3_INTELFPGA, N * max_inc *sizeof(float));

    //std::string kernel_names[]={"test_sgemv_0","test_sgemv_1","test_sgemv_2","test_sgemv_3"};
    std::string kernel_names[2][4]={{"test_sgemv_0","test_sgemv_1","test_sgemv_2","test_sgemv_3"},
                                    {"test_sgemv_4","test_sgemv_5","test_sgemv_6","test_sgemv_7"}};
    int test_case=1;
    GTEST_COUT << "Testing " <<nd*2*nalf*nincs*nbeta<< " cases. This may takes a while."<<std::endl;
    for(int in=0; in<nd;++in)  //Dimension loop
    {
        int n=ndim[in];
        int m=n/2+1;
        //Generate the matrix A
        generate_matrix<float>(A,n,m);
        queue.enqueueWriteBuffer(input_A,CL_TRUE,0,n*m*sizeof(float),A);

        for(int ic = 1; ic <2; ic++) //notrans/trans
        {
            bool trans=(icht[ic]=='T');
            int nl, ml;
            if(trans)
            {
                ml=n;
                nl=m;
            }
            else
            {
                ml=m;
                nl=n;
            }

            for(int ix = 0; ix < nincs; ix ++) //incx and incy
            {

                incx=incxs[ix];
                incy=incys[ix];
                int lx = abs(incx) * ml;
                int ly = abs(incy) * nl;

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

                        queue.enqueueWriteBuffer(input_x,CL_TRUE,0,lx*sizeof(float),x);
                        queue.enqueueWriteBuffer(input_y,CL_TRUE,0,ly*sizeof(float),y);

                        if(!trans)
                            fb.sgemv(kernel_names[ic][ix],FBLAS_NO_TRANSPOSED,n,m,alpha,input_A,m,input_x,incx,beta,input_y,incy);
                        else
                            fb.sgemv(kernel_names[ic][ix],FBLAS_TRANSPOSED,n,m,alpha,input_A,m,input_x,incx,beta,input_y,incy);
                        queue.enqueueReadBuffer(input_y,CL_TRUE,0,ly*sizeof(float),fpga_res);
                        //check
                        check_result<float>(trans, n,m,alpha,A,m,x,incx,beta,y,incy,fpga_res);
                        test_case++;
                    }
                }


            }

        }
    }
    //buffers and other objects are automatically deleted at program exit
}
TEST(TestDgemv,TestDgemv)
{

    const double alphas[nalf] = {0,1.0,0.7};
    const double betas[nbeta] = {0,1.0,0.9};
    cl::CommandQueue queue;
    cl::Context context=fb.get_context();
    cl::Device device=fb.get_device();
    IntelFPGAOCLUtils::createCommandQueue(context,device,queue);
    double *A,*x,*y,*fpga_res;
    posix_memalign ((void **)&A, IntelFPGAOCLUtils::AOCL_ALIGNMENT, N*M*sizeof(double));
    posix_memalign ((void **)&x, IntelFPGAOCLUtils::AOCL_ALIGNMENT, N*max_inc*sizeof(double));
    posix_memalign ((void **)&y, IntelFPGAOCLUtils::AOCL_ALIGNMENT, M*max_inc*sizeof(double));
    posix_memalign ((void **)&fpga_res, IntelFPGAOCLUtils::AOCL_ALIGNMENT, N*max_inc*sizeof(double));

    cl::Buffer input_A(context, CL_MEM_READ_ONLY|CL_CHANNEL_1_INTELFPGA, N * M*sizeof(double));
    cl::Buffer input_x(context, CL_MEM_READ_ONLY|CL_CHANNEL_2_INTELFPGA, N * max_inc* sizeof(double));
    cl::Buffer input_y(context, CL_MEM_READ_WRITE|CL_CHANNEL_3_INTELFPGA, N * max_inc *sizeof(double));
    std::string kernel_names[2][4]={{"test_dgemv_0","test_dgemv_1","test_dgemv_2","test_dgemv_3"},
                                    {"test_dgemv_4","test_dgemv_5","test_dgemv_6","test_dgemv_7"}};
    for(int in=0; in<nd;++in)  //Dimension loop
    {
        int incx=1;
        int incy=1;
        int n=ndim[in];
        int m=n/2+1;

        //Generate the matrix A
        generate_matrix<double>(A,n,m);
        queue.enqueueWriteBuffer(input_A,CL_TRUE,0,n*m*sizeof(double),A);

        for(int ic = 0; ic <2; ic++) //notrans/trans
        {
            bool trans=(icht[ic]=='T');
            int nl, ml;
            if(trans)
            {
                ml=n;
                nl=m;
            }
            else
            {
                ml=m;
                nl=n;
            }

            for(int ix = 0; ix < nincs; ix ++) //incx and incy
            {
                //generate X
                incx=incxs[ix];
                incy=incys[ix];
                int lx = abs(incx) * ml;
                int ly = abs(incy) * nl;

                //loop over alpha and beta
                for(int ia = 0; ia < nalf ; ia++)
                {
                    double alpha=alphas[ia];
                    for(int ib = 0; ib < nbeta; ib++ )
                    {
                        double beta = betas[ib];
                        generate_vector<double>(x,lx);
                        generate_vector<double>(y,ly);

                        queue.enqueueWriteBuffer(input_x,CL_TRUE,0,lx*sizeof(double),x);
                        queue.enqueueWriteBuffer(input_y,CL_TRUE,0,ly*sizeof(double),y);

                        if(!trans)
                            fb.dgemv(kernel_names[ic][ix],FBLAS_NO_TRANSPOSED,n,m,alpha,input_A,m,input_x,incx,beta,input_y,incy);
                        else
                            fb.dgemv(kernel_names[ic][ix],FBLAS_TRANSPOSED,n,m,alpha,input_A,m,input_x,incx,beta,input_y,incy);
                        queue.enqueueReadBuffer(input_y,CL_TRUE,0,ly*sizeof(double),fpga_res);
                        check_result<double>(trans, n,m,alpha,A,m,x,incx,beta,y,incy,fpga_res);
                    }
                }
            }
        }
    }
    //buffers and other objects are automatically deleted at program exit
}
int main(int argc, char *argv[])
{
    if(argc<3)
    {
        std::cerr << "Usage: env CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1 " << argv[0] << " <fpga binary file> <json description>" << std::endl;
        return -1;
    }
    std::string program_path=argv[1];
    std::string json_path=argv[2];
    fb =FBLASEnvironment (program_path,json_path);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
