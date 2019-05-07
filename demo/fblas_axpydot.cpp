/*
        Implementation of  Axydot using FBLAS host API.
        It computes:
            z = w - alpha*v
            beta = z**T*u
        The routines are invoked on FPGA. FPGA code must be generated using
        the host-generator and the fblas_axpydot.json description file.

        CPU computation is enabled by the macro CHECK
*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <string>
#include <fstream>
#include <type_traits>
#include <fblas_environment.hpp>
#include "utils.hpp"
#define CHECK //disable to avoid check
#if defined(CHECK)
#include <cblas.h>
#endif
using namespace std;


template <typename T>
T testFBLAS(std::string program_path, std::string json_path, T* u, T* v, T* w, int n, float alpha, std::vector<double> &times, std::vector <double> &transfer_times,int runs)
{
    FBLASEnvironment fb(program_path,json_path);

    cl::CommandQueue queue;
    cl::Context context=fb.get_context();
    cl::Device device=fb.get_device();
    IntelFPGAOCLUtils::createCommandQueue(context,device,queue);

    timestamp_t comp_start;
    //create data
    T *fpga_res;
    size_t data_size=sizeof(T);
    posix_memalign ((void **)&fpga_res, IntelFPGAOCLUtils::AOCL_ALIGNMENT, data_size);
    cl::Buffer input_u(context, CL_MEM_READ_ONLY|CL_CHANNEL_1_INTELFPGA, n *data_size);
    cl::Buffer input_w(context, CL_MEM_READ_ONLY|CL_CHANNEL_2_INTELFPGA, n * data_size);
    cl::Buffer input_v(context, CL_MEM_READ_ONLY|CL_CHANNEL_3_INTELFPGA, n * data_size);
    cl::Buffer input_z(context, CL_MEM_READ_WRITE|CL_CHANNEL_4_INTELFPGA, n * data_size);
    cl::Buffer output(context, CL_MEM_WRITE_ONLY|CL_CHANNEL_1_INTELFPGA, data_size);
    timestamp_t transf;
    for(int i=0;i<runs;i++)
    {
        comp_start=current_time_usecs();
        queue.enqueueWriteBuffer(input_u,CL_FALSE,0,n*data_size,u);
        queue.enqueueWriteBuffer(input_w,CL_FALSE,0,n*data_size,w);
        queue.enqueueWriteBuffer(input_v,CL_FALSE,0,n*data_size,v);
        queue.finish();
        transf=current_time_usecs()-comp_start;

        asm volatile("": : :"memory");

        comp_start=current_time_usecs();

        //start computation
        fb.scopy("axpydot_fblas_scopy",n,input_w,1,input_z,1);
        fb.saxpy("axpydot_fblas_saxpy",n,-alpha,input_v,1,input_z,1);
        fb.sdot("axpydot_fblas_sdot",n,input_z,1,input_u,1,output);


        asm volatile("": : :"memory");
        //save the computation time
        times.push_back(current_time_usecs()-comp_start);

        //get back the result (and save the time)
        comp_start=current_time_usecs();
        queue.enqueueReadBuffer(output,CL_TRUE,0,data_size,fpga_res);
        transf+=current_time_usecs()-comp_start;
        transfer_times.push_back(transf);
    }

    return *fpga_res;
}

int main(int argc, char *argv[])
{

    //command line argument parsing
    if(argc<11)
    {
        cerr << "Usage: "<< argv[0]<<" -b <binary file fblas> -j <json file>  -n <length of the vectors> -a <alpha>  -r <num runs>"<<endl;
        exit(-1);
    }

    int c;
    int n,runs;
    double alpha;
    std::string program_path;
    std::string json_path;
    while ((c = getopt (argc, argv, "n:j:b:a:r:")) != -1)
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
            case 'j':
                json_path=std::string(optarg);
                break;
            default:
                cerr << "Usage: "<< argv[0]<<" -b <binary file fblas> -j <json file>  -n <length of the vectors> -a <alpha>  -r <num runs>"<<endl;
                exit(-1);
        }
    std::vector<double> fblas_times,streamed_times, transfer_times;

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


    //Fblas
    float fblas_res;
    fblas_res=testFBLAS<float>(program_path,json_path,(float *)u,(float *)v,(float *)w,n,alpha,fblas_times,transfer_times,runs);

    //check
#if defined(CHECK)
    float beta;

    timestamp_t comp_start=current_time_usecs();
        cblas_scopy(n,(float *)w,1,(float *)z,1);
        cblas_saxpy(n,-alpha,(float *)v,1,(float *)z,1);
        beta=cblas_sdot(n,(float *)z,1,(float *)u,1);
    timestamp_t cpu_time=current_time_usecs()-comp_start;


    if(!test_equals(fblas_res,beta,1e-4))
        cout << "Error: " <<fblas_res<<" != " <<beta<<endl;
    else
        cout << "Result is OK"<<endl;
#endif




    //compute the average and standard deviation of times
    double fblas_mean=0;
    for(auto t:fblas_times)
        fblas_mean+=t;
    fblas_mean/=runs;

    double fblas_stddev=0;
    for(auto t:fblas_times)
        fblas_stddev+=((t-fblas_mean)*(t-fblas_mean));
    fblas_stddev=sqrt(fblas_stddev/(runs-1));

    //compute average and standard deviation of transfer times
    double transfer_mean=0, transfer_stddev=0;
    for(auto t:transfer_times)
        transfer_mean+=t;
    transfer_mean/=runs;
    for(auto t:transfer_times)
        transfer_stddev+=((t-transfer_mean)*(t-transfer_mean));
    transfer_stddev=sqrt(transfer_stddev/(runs-1));

    double fblas_conf_interval_99=2.58*fblas_stddev/sqrt(runs);

#if defined(CHECK)
    cout << "Computation time over cpu (usecs): "<<cpu_time<<endl;
#endif
    cout << "Transfer time mesured with FBLAS (usec) " << transfer_mean << " (sttdev: " << transfer_stddev<<")"<<endl;
    cout << "Computation time over fpga with FBLAS (usecs): "<<fblas_mean<< " (sttdev: " << fblas_stddev<<")"<<endl;
    cout << "FBLAS Conf interval 99: "<<fblas_conf_interval_99<<endl;
    cout << "FBLAS Conf interval 99 within " <<(fblas_conf_interval_99/fblas_mean)*100<<"% from mean" <<endl;



}
