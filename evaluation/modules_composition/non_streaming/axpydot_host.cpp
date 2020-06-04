/*
    Implementation of  Axydot using the host API (non-streaming).
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
#include "../../../include/fblas_environment.hpp"

#define CHECK //disable to avoid check
#if defined(CHECK)
#include <cblas.h>
#endif


using namespace std;


template <typename T>
void testNonStreaming(std::string program_path, std::string json_path, int n, T alpha, std::vector<double> &times, int runs)
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

    if(std::is_same<T, float>::value)
        cout << "Single precision version. " <<endl;
    else
        cout << "Double precision version. " <<endl;
    timestamp_t transf;
    for(int i=0;i<runs;i++)
    {
        comp_start=current_time_usecs();
        queue.enqueueWriteBuffer(input_u,CL_TRUE,0,n*data_size,u);
        queue.enqueueWriteBuffer(input_w,CL_TRUE,0,n*data_size,w);
        queue.enqueueWriteBuffer(input_v,CL_TRUE,0,n*data_size,v);
        queue.finish();

        transf=current_time_usecs()-comp_start;

        asm volatile("": : :"memory");

        comp_start=current_time_usecs();
        asm volatile("": : :"memory");
        if(std::is_same<T, float>::value)
        {
            fb.scopy("saxpydot_copy",n,input_w,1,input_z,1);
            fb.saxpy("saxpydot_axpy",n,-alpha,input_v,1,input_z,1);
            fb.sdot("saxpydot_dot",n,input_z,1,input_u,1,output);
        }
        else
        {
            fb.dcopy("daxpydot_copy",n,input_w,1,input_z,1);
            fb.daxpy("daxpydot_axpy",n,(double)-alpha,input_v,1,input_z,1);
            fb.ddot("daxpydot_dot",n,input_z,1,input_u,1,output);
        }

        asm volatile("": : :"memory");
        times.push_back(current_time_usecs()-comp_start);

        comp_start=current_time_usecs();
        queue.enqueueReadBuffer(output,CL_TRUE,0,data_size,fpga_res);

    }
    //check
    #if defined(CHECK)
    T beta;

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

    bool result_verified;
    if (std::is_same<T, float>::value)
        result_verified = test_equals((float)(*fpga_res),beta,1e-4);
    else
        result_verified = test_equals((double)(*fpga_res),beta,1e-6);

    if(!result_verified)
        cout << "Error: " <<*fpga_res<<" != " <<beta<<endl;
    else
        cout << "Result verified!"<< endl;
    #endif


}


int main(int argc, char *argv[])
{

    //command line argument parsing
    if(argc<13)
    {
        cerr << "Usage: "<< argv[0]<<" -b <binary file> -j <json_file> -n <length of the vectors> -a <alpha>  -r <num runs>  -p <precision float/double>"<<endl;
        exit(-1);
    }


    int c;
    int n,runs;
    double alpha;
    std::string program_path, json_path;
    bool double_precision;
    while ((c = getopt (argc, argv, "n:b:a:r:p:j:")) != -1)
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
            case 'j':
                json_path=std::string(optarg);
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
        testNonStreaming<double>(program_path,json_path,n,alpha,streaming_times,runs);
    else
        testNonStreaming<float>(program_path,json_path, n,alpha,streaming_times,runs);



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

    cout << "Computation time over FPGA (usecs): "<<streamed_mean<< " (sttdev: " << streamed_stddev<<")"<<endl;
    cout << "Conf interval 99: "<<streamed_conf_interval_99<<endl;
    cout << "Conf interval 99 within " <<(streamed_conf_interval_99/streamed_mean)*100<<"% from mean" <<endl;

}
