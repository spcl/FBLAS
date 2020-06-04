/*
    Test for  GEMV product: can be single and double precision
    The routine uses data coming from DRAM.
    Since Intel OpenlCL for FPGA does not support automatic interleaving
    between different modules, this is done automatically by striping the data.

*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <string>
#include <chrono>
#include <math.h>
#include "../../include/utils/ocl_utils.hpp"
#include "../../include/utils/utils.hpp"
#include "../../include/utils/test.hpp"
#include "../../include/utils/data_generators.hpp"
#include <sys/time.h>
#define CHECK
#if defined(CHECK)
#include <cblas.h>
#endif

using namespace std;


template <typename T>
void evaluate(std::string program_path,size_t n, size_t m, T alpha, T beta, std::vector<double> &fblas_times, std::vector<double> &transfer_times, int runs){

    cl::Platform  platform;
    cl::Device device;
    cl::Context context;
    cl::Program program;
    std::vector<cl::Kernel> kernels;
    std::vector<cl::CommandQueue> queues;
    std::vector<std::string> kernel_names;
    if (std::is_same<T, float>::value){
        std::cout << "Executing SGEMV ...." <<std::endl;
        kernel_names = {"sgemv", "sgemv_read_x","sgemv_read_y", "sgemv_read_matrix","sgemv_write_vector"};
    }
    else{
        std::cout << "Executing DGEMV ...." <<std::endl;
        kernel_names = {"dgemv", "dgemv_read_x","dgemv_read_y", "dgemv_read_matrix","dgemv_write_vector"};
    }

    IntelFPGAOCLUtils::initEnvironment(platform,device,context,program,program_path,kernel_names, kernels,queues);
    std::cout << "Board reprogrammed." <<std::endl;

    timestamp_t comp_start, comp_time;
    int len_x=m, len_y=n;
    T *A,*A_0, *A_1, *A_2, *A_3,*x,*y,*fpga_res;
    posix_memalign ((void **)&A, IntelFPGAOCLUtils::AOCL_ALIGNMENT, n*m*sizeof(T));
    posix_memalign ((void **)&A_0, IntelFPGAOCLUtils::AOCL_ALIGNMENT, n*m*sizeof(T)/4);
    posix_memalign ((void **)&A_1, IntelFPGAOCLUtils::AOCL_ALIGNMENT, n*m*sizeof(T)/4);
    posix_memalign ((void **)&A_2, IntelFPGAOCLUtils::AOCL_ALIGNMENT, n*m*sizeof(T)/4);
    posix_memalign ((void **)&A_3, IntelFPGAOCLUtils::AOCL_ALIGNMENT, n*m*sizeof(T)/4);

    posix_memalign ((void **)&x, IntelFPGAOCLUtils::AOCL_ALIGNMENT, len_x*sizeof(T));
    posix_memalign ((void **)&y, IntelFPGAOCLUtils::AOCL_ALIGNMENT, len_y*sizeof(T));
    posix_memalign ((void **)&fpga_res, IntelFPGAOCLUtils::AOCL_ALIGNMENT, len_y*sizeof(T));


    generate_matrix<T>(A,n,m);
    generate_vector<T>(x,len_x);
    generate_vector<T>(y,len_y);
    size_t elem_per_module=n*m/4;
    cl::Buffer input_A_0(context, CL_MEM_READ_ONLY|CL_CHANNEL_1_INTELFPGA, elem_per_module*sizeof(T));
    cl::Buffer input_A_1(context, CL_MEM_READ_ONLY|CL_CHANNEL_2_INTELFPGA, elem_per_module*sizeof(T));
    cl::Buffer input_A_2(context, CL_MEM_READ_ONLY|CL_CHANNEL_3_INTELFPGA, elem_per_module*sizeof(T));
    cl::Buffer input_A_3(context, CL_MEM_READ_ONLY|CL_CHANNEL_4_INTELFPGA, elem_per_module*sizeof(T));
    cl::Buffer input_x(context, CL_MEM_READ_ONLY, len_x*sizeof(T));
    cl::Buffer input_output_y(context, CL_MEM_READ_WRITE, len_y *sizeof(T));

    //copy the matrix interleaving it into two modules
    size_t offset=0;
    int width;
    if (std::is_same<T, float>::value)
        width = 64;
    else
        width = 32;
    size_t increment=width/4;
    const int loop_it=((int)(m))/width;   //W must be a divisor of M
    for(int i=0;i<n;i++)
    {
        for(int j=0;j<loop_it;j++)
        {
            //write to the different memory area
            for(int ii=0;ii<16;ii++)
                A_0[offset+ii]=A[i*m+j*width+ii];
            for(int ii=0;ii<16;ii++)
                A_1[offset+ii]=A[i*m+j*width+width/4+ii];
            for(int ii=0;ii<16;ii++)
                A_2[offset+ii]=A[i*m+j*width+width/2+ii];
            for(int ii=0;ii<16;ii++)
                A_3[offset+ii]=A[i*m+j*width+3*width/4+ii];
            offset+=increment;
        }
    }

    //set kernel arguments
    int one=1;
    int zero=0;
    float fzero=0;
    float fone=1;
    int tile_size=2048;  //attention, change this if necessary


    std::cout << "Executing streamed version with width: "<<width << "and tile "<<tile_size<<endl;
    int x_repetitions=ceil((float)(n)/tile_size);

    //gemv
    kernels[0].setArg(0, sizeof(int),&one);
    kernels[0].setArg(1, sizeof(int),&n);
    kernels[0].setArg(2, sizeof(int),&m);
    kernels[0].setArg(3, sizeof(T),&alpha);
    kernels[0].setArg(4, sizeof(T),&beta);

    //read vector_x
    kernels[1].setArg(0, sizeof(cl_mem),&input_x);
    kernels[1].setArg(1, sizeof(int),&n);
    kernels[1].setArg(2, sizeof(int),&tile_size);
    kernels[1].setArg(3, sizeof(int),&x_repetitions);

    //readv vector y
    kernels[2].setArg(0, sizeof(cl_mem),&input_output_y);
    kernels[2].setArg(1, sizeof(int),&n);
    kernels[2].setArg(2, sizeof(int),&tile_size);
    kernels[2].setArg(3, sizeof(int),&one);

    //read matrix a
    kernels[3].setArg(0, sizeof(cl_mem),&input_A_0);
    kernels[3].setArg(1, sizeof(cl_mem),&input_A_1);
    kernels[3].setArg(2, sizeof(cl_mem),&input_A_2);
    kernels[3].setArg(3, sizeof(cl_mem),&input_A_3);
    kernels[3].setArg(4, sizeof(int),&n);
    kernels[3].setArg(5, sizeof(int),&m);
    kernels[3].setArg(6, sizeof(int),&m);

    //write vector
    kernels[4].setArg(0, sizeof(cl_mem),&input_output_y);
    kernels[4].setArg(1, sizeof(int),&n);
    kernels[4].setArg(2, sizeof(int),&tile_size);

    printf("Starting...\n");
    timestamp_t transf;
    for(int i=0;i<runs;i++)
    {
        comp_start=current_time_usecs();

        queues[0].enqueueWriteBuffer(input_A_0, CL_FALSE,0,  n*m*sizeof(T)/4,A_0);
        queues[0].enqueueWriteBuffer(input_A_1, CL_FALSE,0,  n*m*sizeof(T)/4,A_1);
        queues[0].enqueueWriteBuffer(input_A_2, CL_FALSE,0,  n*m*sizeof(T)/4,A_2);
        queues[0].enqueueWriteBuffer(input_A_3, CL_FALSE,0,  n*m*sizeof(T)/4,A_3);
        queues[0].enqueueWriteBuffer(input_x,CL_FALSE,0,m*sizeof(T),x);
        queues[0].enqueueWriteBuffer(input_output_y,CL_FALSE,0,n*sizeof(T),y);
        queues[0].finish();

        transf=current_time_usecs()-comp_start;

        comp_start=current_time_usecs();
        for(int i=0;i<kernel_names.size();i++)
            queues[i].enqueueTask(kernels[i]);
        for(int i=0;i<kernel_names.size();i++)
            queues[i].finish();

        comp_time=current_time_usecs()-comp_start;

        comp_start=current_time_usecs();
        queues[0].enqueueReadBuffer(input_output_y,CL_FALSE,0,n*sizeof(T),fpga_res);
        queues[0].finish();
        transf+=current_time_usecs()-comp_start;
        transfer_times.push_back(transf);
        fblas_times.push_back(comp_time);

    }

#if defined(CHECK)
    double flteps;
    if (std::is_same<T, float>::value)
        flteps = 1e-4;
    else
        flteps = 1e-6;

    if (std::is_same<T, float>::value)
        cblas_sgemv(CblasRowMajor,CblasNoTrans,n,m,(float)alpha,(float *)A,m, (float *)x,1,(float)beta,(float *)y,1);
    else
        cblas_dgemv(CblasRowMajor,CblasNoTrans,n,m,(double)alpha,(double *)A,m,(double *)x,1,(double)beta,(double *)y,1);

    T nrm1_diff=0, nrm1_orig=0;
    T error;

    for(int i=0;i < n; i++)
    {
        nrm1_diff+=abs(fpga_res[i]-y[i]);
        nrm1_orig+=abs(y[i]);
    }
    if(nrm1_diff ==0 && nrm1_orig ==0)
        error=0;
    else
        error=nrm1_diff/nrm1_orig;
    if(error<flteps)
        cout << "Computation OK" << endl;
    else
        cout << "Error = " <<error<<" !!!!" <<endl;
#endif
}
int main(int argc, char *argv[])
{

    //command line argument parsing
    if(argc<15)
    {
        cerr << "Usage: "<< argv[0]<<" -b <binary file> -n <row of A> -m <col of A> -a <alpha> -c <beta> -r <runs> -p <precision float/double>"<<endl;
        exit(-1);
    }

    int c;
    int n,m;
    std::string program_path;
    std::string json_path;
    float alpha,beta;
    int runs;
    bool double_precision;
    while ((c = getopt (argc, argv, "n:b:a:c:m:r:p:")) != -1)
        switch (c)
        {
            case 'n':
                n=atoi(optarg);
                break;
            case 'm':
                m=atoi(optarg);
                break;
            case 'r':
                runs=atoi(optarg);
                break;
            case 'a':
                alpha=atof(optarg);
                break;
            case 'c':
                beta=atof(optarg);
                break;
            case 'b':
                program_path=std::string(optarg);
                break;
            case 'j':
                json_path=std::string(optarg);
                break;
            case 'p':
            {
                std::string str=optarg;
                if(str=="float")
                    double_precision=false;
                else
                    if(str=="double")
                        double_precision=true;
                    else
                    {
                        cerr << "Unrecognized option: " <<optarg<<endl;
                        exit(-1);
                    }
                break;
            }
            default:
                cerr << "Usage: "<< argv[0]<<" -b <binary file> -n <row of A> -m <col of A> -a <alpha> -c <beta> -r <runs> -p <precision float/double>"<<endl;
                exit(-1);
        }
    std::vector<double> fblas_times,transfer_times;
    timestamp_t comp_time;

    if(!double_precision)
        evaluate<float>(program_path,n,m,alpha, beta, fblas_times,transfer_times,runs);
    else
        evaluate<double>(program_path,n,m,alpha, beta, fblas_times,transfer_times,runs);


    double mean=0, mean_transf;
    for(auto t:fblas_times)
        mean+=t;
    mean/=runs;
    for(auto t:transfer_times)
        mean_transf+=t;
    mean_transf/=runs;
    //report the mean in usecs

    double stddev=0;
    for(auto t:fblas_times)
        stddev+=((t-mean)*(t-mean));
    stddev=sqrt(stddev/runs);
    cout << "Computation time over fpga (usecs): "<<mean<<endl;
    cout << "Transfer time (usec): " << mean_transf << std::endl;

    double gops=(((double)(2.0f*n*m+2*n))/(1000000000.0));
    std::cout << "FPGA GOps/s: " << gops/((mean)/1000000.0)<<std::endl;

    double data_bytes=(double_precision)? (2*n+n*m)*sizeof(double): (2*n+n*m)*sizeof(float); //2 times because you have to read and to write

    double comp_bandwidth=((double)data_bytes/(mean/1000000.0))/(1024*1024*1024); //GB/s
    cout << "Computation bandwidth: " << comp_bandwidth << " GB/s"<<endl;

}
