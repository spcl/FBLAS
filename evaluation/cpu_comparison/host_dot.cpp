/**
    Test for  DOT product: can be single and double precision
    The routine uses data coming from DRAM.
    Since Intel OpenlCL for FPGA does not support automatic interleaving
    between different modules, this is done automatically by striping the data.

*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <string>
#include <cblas.h>
#include <cassert>
#include "../../include/utils/ocl_utils.hpp"
#include "../../include/utils/utils.hpp"
#include "../../include/utils/test.hpp"
#include "../../include/utils/data_generators.hpp"
using namespace std;

template <typename T>
void evaluate(std::string program_path,int n,  std::vector<double> &fblas_times, std::vector<double> &transfer_times, int runs){
    cl::Platform  platform;
    cl::Device device;
    cl::Context context;
    cl::Program program;
    std::vector<cl::Kernel> kernels;
    std::vector<cl::CommandQueue> queues;
    std::vector<std::string> kernel_names;
    if (std::is_same<T, float>::value)
        kernel_names = {"sdot", "read_vector_x","read_vector_y", "sdot_write_result"};
    else
        kernel_names = {"ddot", "read_vector_x","read_vector_y", "ddot_write_result"};

    IntelFPGAOCLUtils::initEnvironment(platform,device,context,program,program_path,kernel_names, kernels,queues);


    timestamp_t comp_start,comp_time;
    T *x,*y,*fpga_res;
    posix_memalign ((void **)&x, IntelFPGAOCLUtils::AOCL_ALIGNMENT, n*sizeof(T));
    posix_memalign ((void **)&y, IntelFPGAOCLUtils::AOCL_ALIGNMENT, n*sizeof(T));
    posix_memalign ((void **)&fpga_res, IntelFPGAOCLUtils::AOCL_ALIGNMENT, sizeof(T));

    if (std::is_same<T, float>::value){
        generate_float_vector((float *)x,n);
        generate_float_vector((float *)y,n);
    }
    else{
        generate_double_vector((double *)x,n);
        generate_double_vector((double *)y,n);
    }
    size_t elem_per_module=n/2;
    cl::Buffer input_x_0(context, CL_MEM_READ_ONLY|CL_CHANNEL_1_INTELFPGA, elem_per_module*sizeof(T));
    cl::Buffer input_x_1(context, CL_MEM_READ_ONLY|CL_CHANNEL_2_INTELFPGA, elem_per_module*sizeof(T));
    cl::Buffer input_y_0(context, CL_MEM_READ_ONLY|CL_CHANNEL_3_INTELFPGA, elem_per_module*sizeof(T));
    cl::Buffer input_y_1(context, CL_MEM_READ_ONLY|CL_CHANNEL_4_INTELFPGA, elem_per_module*sizeof(T));
    cl::Buffer output(context, CL_MEM_READ_WRITE, sizeof(T));

    //copy the matrix interleaving it into two modules
    size_t offset=0;
    int width;
    if (std::is_same<T, float>::value)
       width=32;
    else
       width=16;
    assert(n%width==0);
    int loop_it=((int)(n))/width; //n must be a multiple

    //prepare data
    T *x_0,*x_1,*y_0,*y_1;
    posix_memalign ((void **)&x_0, IntelFPGAOCLUtils::AOCL_ALIGNMENT, n/2*sizeof(T));
    posix_memalign ((void **)&x_1, IntelFPGAOCLUtils::AOCL_ALIGNMENT, n/2*sizeof(T));
    posix_memalign ((void **)&y_0, IntelFPGAOCLUtils::AOCL_ALIGNMENT, n/2*sizeof(T));
    posix_memalign ((void **)&y_1, IntelFPGAOCLUtils::AOCL_ALIGNMENT, n/2*sizeof(T));


    for(int j=0;j<loop_it;j++)
    {
        //write to the different banks
        for(int ii=0;ii<width/2;ii++)
            x_0[j*width/2+ii] = x[j*width+ii];
        for(int ii=0;ii<width/2;ii++)
            x_1[j*width/2+ii] = x[j*width+width/2+ii];
        for(int ii=0;ii<width/2;ii++)
            y_0[j*width/2+ii] = y[j*width+ii];
        for(int ii=0;ii<width/2;ii++)
            y_1[j*width/2+ii] = y[j*width+width/2+ii];
    }

    int one=1;
    int zero=0;
    std::cout << "Executing streamed version with width: "<<width <<endl;

    //dot
    kernels[0].setArg(0, sizeof(int),&n);

    //read vector_x
    kernels[1].setArg(0, sizeof(cl_mem),&input_x_0);
    kernels[1].setArg(1, sizeof(cl_mem),&input_x_1);
    kernels[1].setArg(2, sizeof(int),&n);
    kernels[1].setArg(3, sizeof(int),&width);
    kernels[1].setArg(4, sizeof(int),&one);

    //readv vector y
    kernels[2].setArg(0, sizeof(cl_mem),&input_y_0);
    kernels[2].setArg(1, sizeof(cl_mem),&input_y_1);
    kernels[2].setArg(2, sizeof(int),&n);
    kernels[2].setArg(3, sizeof(int),&width);
    kernels[2].setArg(4, sizeof(int),&one);

    //write
    kernels[3].setArg(0, sizeof(cl_mem),&output);

    for(int i=0;i<runs;i++)
    {
        timestamp_t transf=current_time_usecs();
        queues[0].enqueueWriteBuffer(input_x_0, CL_FALSE,0, (n/2)*sizeof(T),x_0);
        queues[0].enqueueWriteBuffer(input_x_1, CL_FALSE,0, (n/2)*sizeof(T),x_1);
        queues[0].enqueueWriteBuffer(input_y_0, CL_FALSE,0, (n/2)*sizeof(T),y_0);
        queues[0].enqueueWriteBuffer(input_y_1, CL_FALSE,0, (n/2)*sizeof(T),y_1);
        queues[0].finish();
        transf = current_time_usecs()-transf;


        comp_start=current_time_usecs();
        for(int i=0;i<kernel_names.size();i++)
            queues[i].enqueueTask(kernels[i]);
        for(int i=0;i<kernel_names.size();i++)
            queues[i].finish();

        comp_time=current_time_usecs()-comp_start;


        comp_start=current_time_usecs();
        queues[0].enqueueReadBuffer(output,CL_TRUE,0,sizeof(T),fpga_res);
        transf += current_time_usecs()-comp_start;
        transfer_times.push_back(transf);
        fblas_times.push_back(comp_time);
    }
    T cblas_res;
    if (std::is_same<T, float>::value)
        cblas_res = cblas_sdot(n,(float *)x,1,(float *)y,1);
    else
        cblas_res = cblas_ddot(n,(double *)x,1,(double *)y,1);

    if(!test_equals(*fpga_res,cblas_res,flteps))
        cout << "Error: " <<*fpga_res<<" != " <<cblas_res<<endl;
    else
        cout << "Result Ok!"<<*fpga_res<<" != " <<cblas_res<<endl;;

}


int main(int argc, char *argv[])
{

    //command line argument parsing
    if(argc<9)
    {
        cerr << "Usage: "<< argv[0]<<" -b <binary file> -n <length of the vectors> -r <runs> -p <precision float/double>"<<endl;
        exit(-1);
    }

    int c;
    int n;
    int incx,incy;
    bool double_precision;
    std::string program_path;
    std::string json_path;
    int runs=1;
    while ((c = getopt (argc, argv, "n:b:r:p:")) != -1)
        switch (c)
        {
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
            case 'n':
                n=atoi(optarg);
                break;
            case 'b':
                program_path=std::string(optarg);
                break;
            case 'r':
                runs = atoi(optarg);
                break;
            default:
                cerr << "Usage: "<< argv[0]<<" -b <binary file> -n <length of the vectors> -r <runs> -p <precision float/double>"<<endl;
                exit(-1);
        }


    std::vector<double> fblas_times,transfer_times;
    if(!double_precision)
        evaluate<float>(program_path,n,fblas_times,transfer_times,runs);
    else
        evaluate<double>(program_path,n,fblas_times,transfer_times,runs);



    double gops=((double)2.0*n)/1000000000;

    double data_bytes=(double_precision)?2*n*sizeof(double):2*n*sizeof(float);

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

    double computation_gops=(2.0f*n)/1000000000;
    double measured_gops=computation_gops/((mean)/1000000.0);
    cout << "FPGA Computation time (usec): " << mean << " (sttdev: " << stddev<<")"<<endl;
    std::cout << "FPGA GOps/s: " << measured_gops<<std::endl;
    double comp_bandwidth=((double)data_bytes/(mean/1000000.0))/(1024*1024*1024); //GB/s
    cout << "Transfer time (usec): " << mean_transf << std::endl;
}
