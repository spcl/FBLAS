/*
    Implementation of  Axydot using the module composition.
    It computes:
        z = w - alpha*v
        beta = z**T*u
    The result of the first AXPY is streamed directly towards the dot product.

    The routines are invoked on FPGA. FPGA code must be generated using
    the modules-generator and the _axpydot.json description file.

    CPU computation is enabled by the macro CHECK
*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <string>
#include <fstream>
#include <type_traits>
#include "../../../include/utils/ocl_utils.hpp"
#include "../../../include/utils/utils.hpp"
#include "../../../include/utils/test.hpp"
#include "../../../include/utils/data_generators.hpp"
#define CHECK //disable to avoid check
#if defined(CHECK)
#include <cblas.h>
#endif


using namespace std;


template <typename T>
void testStreaming(std::string program_path, int n, T alpha, std::vector<double> &times, int runs)
{
    cl::Platform platform;
    cl::Device device;
    cl::Context context;
    cl::Program program;
    IntelFPGAOCLUtils::initOpenCL(platform,device,context,program,program_path);
    std::vector<std::string> kernel_names;
    if (std::is_same<T, float>::value){
        cout << "Executing in single precision..." <<std::endl;
        kernel_names={"saxpydot_axpy","saxpydot_read_v", "saxpydot_read_w", "saxpydot_dot", "saxpydot_read_u", "saxpydot_write_beta"};
    }
    else{
        cout << "Executing in double precision..." << std::endl;
        kernel_names={"daxpydot_axpy","daxpydot_read_v", "daxpydot_read_w", "daxpydot_dot", "daxpydot_read_u", "daxpydot_write_beta"};
    }
    std::vector<cl::Kernel> kernels;
    std::vector<cl::CommandQueue> queues;
    const int num_kernels=kernel_names.size();
    IntelFPGAOCLUtils::initEnvironment(platform,device,context,program,program_path,kernel_names, kernels,queues);
    cout << "Board reprogrammed." << endl;
    T fpga_res_streamed;
    size_t data_size=sizeof(T);

    void * u,*w,*v,*z;
    //create data

    posix_memalign ((void **)&u, IntelFPGAOCLUtils::AOCL_ALIGNMENT, n*data_size);
    posix_memalign ((void **)&w, IntelFPGAOCLUtils::AOCL_ALIGNMENT, n*data_size);
    posix_memalign ((void **)&v, IntelFPGAOCLUtils::AOCL_ALIGNMENT, n*data_size);
    posix_memalign ((void **)&z, IntelFPGAOCLUtils::AOCL_ALIGNMENT, n*data_size);
    srand(time(NULL));
    generate_vector<T>((T *)u,n);
    generate_vector<T>((T *)w,n);
    generate_vector<T>((T *)v,n);


    unsigned int width;
     //width used in the implementation. Please change accordingly
    if (std::is_same<T, float>::value)
        width = 16;
    else
        width = 8;

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


    queues[0].enqueueReadBuffer(output_streamed,CL_TRUE,0,data_size,&fpga_res_streamed);

    //check
    #if defined(CHECK)
    T beta;

    timestamp_t comp_start=current_time_usecs();
    if (std::is_same<T, float>::value){
        cblas_scopy(n,(float *)w,1,(float *)z,1);
        cblas_saxpy(n,-alpha,(float *)v,1,(float *)z,1);
        beta=cblas_sdot(n,(float *)z,1,(float *)u,1);
    }
    else{
        cblas_dcopy(n,(double *)w,1,(double *)z,1);
        cblas_daxpy(n,-alpha,(double *)v,1,(double *)z,1);
        beta=cblas_ddot(n,(double *)z,1,(double *)u,1);
    }
    timestamp_t cpu_time=current_time_usecs()-comp_start;

    bool result_verified;
    if (std::is_same<T, float>::value)
        result_verified = test_equals(fpga_res_streamed,beta,1e-4);
    else
        result_verified = test_equals(fpga_res_streamed,beta,1e-6);

    if(!result_verified)
        cout << "Error: " <<fpga_res_streamed<<" != " <<beta<<endl;
    else
        cout << "Result verified!"<< endl;
    #endif


}


int main(int argc, char *argv[])
{

    //command line argument parsing
    if(argc<11)
    {
        cerr << "Usage: "<< argv[0]<<" -b <binary file> -n <length of the vectors> -a <alpha>  -r <num runs>  -p <precision float/double>"<<endl;
        exit(-1);
    }


    int c;
    int n,runs;
    double alpha;
    std::string program_path;
    bool double_precision;
    while ((c = getopt (argc, argv, "n:b:a:r:p:")) != -1)
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
                cerr << "Usage: "<< argv[0]<<" -b <binary file fblas> -n <length of the vectors> -a <alpha>  -r <num runs>  -p <precision float/double>"<<endl;
                exit(-1);
        }
    
    std::vector<double> streaming_times;
    
    if(double_precision)
        testStreaming<double>(program_path,n,alpha,streaming_times,runs);
    else
        testStreaming<float>(program_path,n,alpha,streaming_times,runs);

    

    //compute the average and standard deviation of times
    double streamed_mean=0;
    for(auto t:streaming_times)
        streamed_mean+=t;
    streamed_mean/=runs;

    double streamed_stddev=0;
    for(auto t:streaming_times)
        streamed_stddev+=((t-streamed_mean)*(t-streamed_mean));
    streamed_stddev=sqrt(streamed_stddev/(runs-1));

    double streamed_conf_interval_99=2.58*streamed_stddev/sqrt(runs);

    cout << "Computation time over fpga with streamed (usecs): "<<streamed_mean<< " (sttdev: " << streamed_stddev<<")"<<endl;
    cout << "Streamed Conf interval 99: "<<streamed_conf_interval_99<<endl;
    cout << "Streamed Conf interval 99 within " <<(streamed_conf_interval_99/streamed_mean)*100<<"% from mean" <<endl;

}
