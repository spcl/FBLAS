/*
        Implementation of  Axydot using the module composition.
        It computes:
            z = w - alpha*v
            beta = z**T*u
        The result of the first AXPY is streamed directly towards the dot product.

        The routines are invoked on FPGA. FPGA code must be generated using
        the modules-generator and the streamed_axpydot.json description file.

        CPU computation is enabled by the macro CHECK
*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <string>
#include <fstream>
#include <type_traits>
#include <utils/ocl_utils.hpp>
#include "utils.hpp"
#define CHECK //disable to avoid check
#if defined(CHECK)
#include <cblas.h>
#endif

unsigned int width=32; //width used in the implementation

using namespace std;


template <typename T>
T testStreamed(std::string program_path, T* u, T* v, T *w,int n, float alpha, std::vector<double> &times, int runs)
{
    cl::Platform platform;
    cl::Device device;
    cl::Context context;
    cl::Program program;
    IntelFPGAOCLUtils::initOpenCL(platform,device,context,program,program_path);
    std::vector<std::string> kernel_names={"axpydot_streamed_saxpy","axpydot_streamed_read_vector_v", "axpydot_streamed_read_vector_w", "axpydot_streamed_sdot", "axpydot_streamed_read_vector_u", "axpydot_streamed_write_scalar"};
    std::vector<cl::Kernel> kernels;
    std::vector<cl::CommandQueue> queues;
    const int num_kernels=kernel_names.size();
    IntelFPGAOCLUtils::createCommandQueues(context,device,queues,num_kernels);
    IntelFPGAOCLUtils::createKernels(program,kernel_names,kernels);
    T fpga_res_streamed;
    size_t data_size=sizeof(T);

    //copy back the data (data is erased after reprogramming)
    cl::Buffer input_u(context, CL_MEM_READ_ONLY|CL_CHANNEL_1_INTELFPGA, n *data_size);
    cl::Buffer input_w(context, CL_MEM_READ_ONLY|CL_CHANNEL_2_INTELFPGA, n * data_size);
    cl::Buffer input_v(context, CL_MEM_READ_ONLY|CL_CHANNEL_3_INTELFPGA, n * data_size);
    cl::Buffer output_streamed(context, CL_MEM_WRITE_ONLY|CL_CHANNEL_4_INTELFPGA, data_size);

    queues[0].enqueueWriteBuffer(input_u,CL_TRUE,0,n*data_size,u);
    queues[0].enqueueWriteBuffer(input_w,CL_TRUE,0,n*data_size,w);
    queues[0].enqueueWriteBuffer(input_v,CL_TRUE,0,n*data_size,v);
    int one=1;
    //set args
    T al=-alpha;
    kernels[0].setArg(0,data_size,&al);
    kernels[0].setArg(1,sizeof(int),&n);

    kernels[1].setArg(0,sizeof(cl_mem),&input_v);
    kernels[1].setArg(1,sizeof(unsigned int),&n);
    kernels[1].setArg(2,sizeof(unsigned int),&width);
    kernels[1].setArg(3,sizeof(unsigned int),&one);

    kernels[2].setArg(0,sizeof(cl_mem),&input_w);
    kernels[2].setArg(1,sizeof(unsigned int),&n);
    kernels[2].setArg(2,sizeof(unsigned int),&width);
    kernels[2].setArg(3,sizeof(unsigned int),&one);

    kernels[3].setArg(0,sizeof(unsigned int),&n);

    kernels[4].setArg(0,sizeof(cl_mem),&input_u);
    kernels[4].setArg(1,sizeof(unsigned int),&n);
    kernels[4].setArg(2,sizeof(unsigned int),&width);
    kernels[4].setArg(3,sizeof(unsigned int),&one);

    kernels[5].setArg(0,sizeof(cl_mem),&output_streamed);

    //execute and take computation time
    for(int i=0;i<runs;i++)
    {
        timestamp_t comp_start=current_time_usecs();
        asm volatile("": : :"memory");
        for(int i=0;i<num_kernels;i++)
            queues[i].enqueueTask(kernels[i]);

        //wait
        for(int i=0;i<num_kernels;i++)
            queues[i].finish();
        asm volatile("": : :"memory");
        times.push_back(current_time_usecs()-comp_start);
    }

    //check

    queues[0].enqueueReadBuffer(output_streamed,CL_TRUE,0,data_size,&fpga_res_streamed);
    return fpga_res_streamed;
}


int main(int argc, char *argv[])
{

    //command line argument parsing
    if(argc<9)
    {
        cerr << "Usage: "<< argv[0]<<" -b <binary file> -n <length of the vectors> -a <alpha>  -r <num runs>"<<endl;
        exit(-1);
    }

    int c;
    int n,runs;
    double alpha;
    std::string program_path;
    while ((c = getopt (argc, argv, "n:b:a:r:")) != -1)
        switch (c)
        {
            case 'n':
                n=atoi(optarg);
                break;
            case 'a':
                alpha=atof(optarg);
                break;
            case 'b':
                program_path=std::string(optarg);
                break;
            case 'r':
                runs=atoi(optarg);
                break;
            default:
                cerr << "Usage: "<< argv[0]<<" -b <binary file fblas> -n <length of the vectors> -a <alpha>  -r <num runs>"<<endl;
                exit(-1);
        }
    std::vector<double> streamed_times;

    //set affinity
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);

    if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset)) {
        cerr << "Cannot set thread to CPU " << 0<< endl;
    }
    void * u,*w,*v,*z;
    //create data

    posix_memalign ((void **)&u, IntelFPGAOCLUtils::AOCL_ALIGNMENT, n*sizeof(float));
    posix_memalign ((void **)&w, IntelFPGAOCLUtils::AOCL_ALIGNMENT, n*sizeof(float));
    posix_memalign ((void **)&v, IntelFPGAOCLUtils::AOCL_ALIGNMENT, n*sizeof(float));
    posix_memalign ((void **)&z, IntelFPGAOCLUtils::AOCL_ALIGNMENT, n*sizeof(float));
    generate_float_vector((float *)u,n);
    generate_float_vector((float *)w,n);
    generate_float_vector((float *)v,n);



    float fpga_res_streamed=testStreamed<float>(program_path,(float *)u,(float *)v,(float *)w,n,alpha,streamed_times,runs);

#if defined(CHECK)
    float beta;

    timestamp_t comp_start=current_time_usecs();
        cblas_scopy(n,(float *)w,1,(float *)z,1);
        cblas_saxpy(n,-alpha,(float *)v,1,(float *)z,1);
        beta=cblas_sdot(n,(float *)z,1,(float *)u,1);
    timestamp_t cpu_time=current_time_usecs()-comp_start;
    if(!test_equals(fpga_res_streamed,beta,1e-4))
        cout << "Error: " <<fpga_res_streamed<<" != " <<beta<<endl;
    else
        cout << "Streamed OK"<<endl;
#endif
    //compute the average and standard deviation of times
    double streamed_mean=0;
    for(auto t:streamed_times)
        streamed_mean+=t;
    streamed_mean/=runs;

    double streamed_stddev=0;
    for(auto t:streamed_times)
        streamed_stddev+=((t-streamed_mean)*(t-streamed_mean));
    streamed_stddev=sqrt(streamed_stddev/(runs-1));

    double streamed_conf_interval_99=2.58*streamed_stddev/sqrt(runs);

#if defined(CHECK)
    cout << "Computation time over cpu (usecs): "<<cpu_time<<endl;
#endif
    cout << "Computation time over fpga with streamed (usecs): "<<streamed_mean<< " (sttdev: " << streamed_stddev<<")"<<endl;
    cout << "Streamed Conf interval 99: "<<streamed_conf_interval_99<<endl;
    cout << "Streamed Conf interval 99 within " <<(streamed_conf_interval_99/streamed_mean)*100<<"% from mean" <<endl;

}
