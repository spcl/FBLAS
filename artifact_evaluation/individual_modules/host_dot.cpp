/*
  Test for  DOT product: can be single and double precision

  The kernels are executed multiple times:
    - for each execution it takes the time using the OpenCL Profiling info command, considering
        the start and the end of a kernel. The execution time is takes that elapses between the start
        of the first kernel to the end of the last one.
    - it outputs in a file all the summary metrics as well as all the measured timings
 */


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string>
#include <iostream>
#include <vector>
#include <cblas.h>
#include "CL/opencl.h"
#include "../../include/utils/ocl_utils.hpp"
#include "../../include/utils/utils.hpp"
#include "../../include/utils/test.hpp"

using namespace std;
#define CHECK //enable to perform check


void singlePrecision(int n,  std::string program_path,std::vector<double> &times, int runs){
    cout << "-------------------------------------------------------------------------"<<endl;
    cout << "Executing single precision streaming dot product with vectors of length: "<<n<<endl;
    cout << "-------------------------------------------------------------------------"<<endl;

    //opencl variables
    cl::Platform platform;
    cl::Device device;
    cl::Context context;
    cl::Program program;
    std::vector<std::string> kernel_names={"sdot_generate_x","sdot_generate_y","sdot","sdot_write_result"};
    std::vector<cl::Kernel> kernels;
    std::vector<cl::CommandQueue> queues;
    IntelFPGAOCLUtils::initEnvironment(platform,device,context,program,program_path,kernel_names,kernels,queues);
    const int num_kernels=kernel_names.size();


    // Output buffer
    cl::Buffer output(context, CL_MEM_WRITE_ONLY,sizeof(float));

    // Set kernel arguments.
    int one=1;

    //generate x
    kernels[0].setArg(0,sizeof(int),&n);
    kernels[0].setArg(1,sizeof(int),&one);

    //generate_y
    kernels[1].setArg(0,sizeof(int),&n);
    kernels[1].setArg(1,sizeof(int),&one);

    //sdot
    kernels[2].setArg(0,sizeof(int),&n);

    //write_retuls
    kernels[3].setArg(0,sizeof(cl_mem),&output);



    //Launch the computation multiple times
    for(int r=0;r<runs;r++)
    {
        cl::Event events[4];

        for(int i=0;i<num_kernels;i++)
        {
           queues[i].enqueueTask(kernels[i],nullptr,&events[i]);
           queues[i].flush();
        }

        //wait
        for(int i=0;i<num_kernels;i++)
           queues[i].finish();


        //compute execution time using OpenCL profiling
        ulong min_start, max_end=0;
        ulong end;
        ulong start;
        for(int i=0;i<num_kernels;i++)
        {
            events[i].getProfilingInfo<ulong>(CL_PROFILING_COMMAND_START,&start);
            events[i].getProfilingInfo<ulong>(CL_PROFILING_COMMAND_END,&end);
            if(i==0)
                min_start=start;
            if(start<min_start)
                min_start=start;
            if(end>max_end)
                max_end=end;
        }
        times.push_back((double)((max_end-min_start)/1000.0f));
    }
    //get back the result

    #if defined(CHECK)
        float *res;
        posix_memalign((void **)&res, IntelFPGAOCLUtils::AOCL_ALIGNMENT, sizeof(float));
        queues[0].enqueueReadBuffer(output,CL_TRUE,0,sizeof(float),res);
        float *x=new float[n];
        float *y=new float[n];
            for(int i=0;i<n;i++)
            {
                x[i]=i;
                y[i]=i;
            }
        float cblas_res= cblas_sdot(n,x,1,y,1);

        if(!test_equals(*res,cblas_res,flteps))
        cout << "Error! " <<*res<<" != " <<cblas_res<<endl;
        else
        cout << "Result Ok!" <<endl;
    #endif



}
void doublePrecision(int n, std::string program_path, std::vector<double> &times , int runs )
{
    cout << "-------------------------------------------------------------------------"<<endl;
    cout << "Executing double precision streaming dot product with vectors of length: "<<n<<endl;
    cout << "-------------------------------------------------------------------------"<<endl;

    //opencl variables
    cl::Platform platform;
    cl::Device device;
    cl::Context context;
    cl::Program program;
    std::vector<std::string> kernel_names={"ddot_generate_x","ddot_generate_y","ddot","ddot_write_result"};
    std::vector<cl::Kernel> kernels;
    std::vector<cl::CommandQueue> queues;
    IntelFPGAOCLUtils::initEnvironment(platform,device,context,program,program_path,kernel_names,kernels,queues);
    const int num_kernels=kernel_names.size();

    // Output buffer

    cl::Buffer output(context, CL_MEM_WRITE_ONLY,sizeof(double));

    // Set kernel arguments.
    int one=1;

    //generate x
    kernels[0].setArg(0,sizeof(int),&n);
    kernels[0].setArg(1,sizeof(int),&one);

    //generate_y
    kernels[1].setArg(0,sizeof(int),&n);
    kernels[1].setArg(1,sizeof(int),&one);

    //sdot
    kernels[2].setArg(0,sizeof(int),&n);

    //write_retuls
    kernels[3].setArg(0,sizeof(cl_mem),&output);

    //Launch the computation multiple times
    for(int r=0;r<runs;r++)
    {
        cl::Event events[4];

        for(int i=0;i<num_kernels;i++)
        {
           queues[i].enqueueTask(kernels[i],nullptr,&events[i]);
           queues[i].flush();
        }

        //wait
        for(int i=0;i<num_kernels;i++)
           queues[i].finish();


        //compute execution time using OpenCL profiling
        ulong min_start, max_end=0;
        ulong end;
        ulong start;
        for(int i=0;i<num_kernels;i++)
        {
            events[i].getProfilingInfo<ulong>(CL_PROFILING_COMMAND_START,&start);
            events[i].getProfilingInfo<ulong>(CL_PROFILING_COMMAND_END,&end);
            if(i==0)
                min_start=start;
            if(start<min_start)
                min_start=start;
            if(end>max_end)
                max_end=end;
        }
        times.push_back((double)((max_end-min_start)/1000.0f));
    }


    #if defined(CHECK)
        double *res;
        posix_memalign((void **)&res, IntelFPGAOCLUtils::AOCL_ALIGNMENT, sizeof(double));
        //get back the result
        queues[0].enqueueReadBuffer(output,CL_TRUE,0,sizeof(double),res);
        double *x=new double[n];
        double *y=new double[n];
            for(int i=0;i<n;i++)
            {
                x[i]=i;
                y[i]=i;
            }
        double cblas_res= cblas_ddot(n,x,1,y,1);

        if(!test_equals(*res,cblas_res,flteps))
            cout << "Error: " <<res<<" != " <<cblas_res<<endl;
        else
            cout << "Result Ok!" <<endl;

    #endif
}



int main(int argc, char *argv[])
{
    //command line argument parsing
    if(argc<9)
    {
        cerr << "Usage: "<< argv[0]<<"-b <binary file> -n <length of the vectors> -p <single/double> -r <num runs>"<<endl;
        exit(-1);
    }

    int c;
    int n, runs;
    bool double_precision;
    std::string program_name;
    while ((c = getopt (argc, argv, "n:p:b:r:")) != -1)
        switch (c)
        {
            case 'n':
                n=atoi(optarg);
                break;
            case 'r':
                runs=atoi(optarg);
                break;
            case 'b':
                program_name=std::string(optarg);
                break;
            case 'p':
                {
                    std::string str=optarg;
                    if(str=="single")
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
                cerr << "Usage: "<< argv[0]<<" -n <length of the vectors> -p <single/double>"<<endl;
                exit(-1);
        }
    std::vector<double> times;
    long data_bytes;
    if(double_precision)
        doublePrecision(n,program_name,times,runs);
    else
        singlePrecision(n,program_name,times,runs);

    //compute the average and standard deviation of times
    double mean=0;
    for(auto t:times)
        mean+=t;
    mean/=runs;
    //report the mean in usecs

    double stddev=0;
    for(auto t:times)
        stddev+=((t-mean)*(t-mean));
    stddev=sqrt(stddev/runs);

    double computation_gops=(2.0f*n)/1000000000;
    double measured_gops=computation_gops/((mean)/1000000.0);
    cout << "FPGA Computation time (usec): " << mean << " (sttdev: " << stddev<<")"<<endl;
    std::cout << "FPGA GOps/s: " << measured_gops<<std::endl;


    return 0;
}
