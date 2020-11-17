/**
 *  BICG, non streamed version
 *      q = A*p
 *      s = A**T*r
 *  A is an NxM matrix. r and q are N elements vector, p  and s are M elements vector
 *  Implementing this with classical BLAS, requires two gemv that can run in parallel (by sharing memory accesses)
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <string>
#include <cblas.h>
#include <sstream>
#include <type_traits>

#include "../../../include/utils/ocl_utils.hpp"
#include "../../../include/utils/utils.hpp"
#include "../../../include/utils/test.hpp"
#include "../../../include/utils/data_generators.hpp"
#include "../../../include/fblas_environment.hpp"

using namespace std;
#define CHECK



template <typename T>
void testNonStreamed(std::string program_path,std::string json_path, int n, int m, std::vector<double> &times, std::vector <double> &transfer_times, timestamp_t &cpu_time,int runs)
{


    FBLASEnvironment fb(program_path,json_path);

    cl::CommandQueue queue;
    cl::Context context=fb.get_context();
    cl::Device device=fb.get_device();
    IntelFPGAOCLUtils::createCommandQueue(context,device,queue);

    timestamp_t comp_start;
    //create data

    T fpga_res_streamed;
    void *fpga_res_q, *fpga_res_s;
    void *streamed_res_q, *streamed_res_s;
    void *A,*p,*r,*q,*s;

     //create data
    posix_memalign ((void **)&A, IntelFPGAOCLUtils::AOCL_ALIGNMENT, n*m*sizeof(T));
    posix_memalign ((void **)&p, IntelFPGAOCLUtils::AOCL_ALIGNMENT, m*sizeof(T));
    posix_memalign ((void **)&q, IntelFPGAOCLUtils::AOCL_ALIGNMENT, n*sizeof(T));
    posix_memalign ((void **)&s, IntelFPGAOCLUtils::AOCL_ALIGNMENT, m*sizeof(T));
    posix_memalign ((void **)&r, IntelFPGAOCLUtils::AOCL_ALIGNMENT, n*sizeof(T));
    posix_memalign ((void **)&fpga_res_q, IntelFPGAOCLUtils::AOCL_ALIGNMENT, n*sizeof(T));
    posix_memalign ((void **)&fpga_res_s, IntelFPGAOCLUtils::AOCL_ALIGNMENT, m*sizeof(T));
    generate_vector<T>((T *)p,m);
    generate_vector<T>((T *)r,n);
    generate_matrix<T>((T *)A,n,m);


    cl::Buffer input_p(context, CL_MEM_READ_ONLY|CL_CHANNEL_1_INTELFPGA, m *sizeof(T));
    cl::Buffer input_r(context, CL_MEM_READ_ONLY|CL_CHANNEL_2_INTELFPGA, n * sizeof(T));
    cl::Buffer input_A(context, CL_MEM_READ_ONLY|CL_CHANNEL_3_INTELFPGA, n * m*sizeof(T));
    cl::Buffer output_q(context, CL_MEM_WRITE_ONLY|CL_CHANNEL_4_INTELFPGA, n * sizeof(T));
    cl::Buffer output_s(context, CL_MEM_WRITE_ONLY|CL_CHANNEL_1_INTELFPGA, m * sizeof(T));

    timestamp_t transfer_time;

    if(std::is_same<T, float>::value)
        cout << "Single precision version. " <<endl;
    else
        cout << "Double precision version. " <<endl;

    for(int i=0;i<runs;i++)
    {
        comp_start=current_time_usecs();
        queue.enqueueWriteBuffer(input_p,CL_FALSE,0,m*sizeof(T),p);
        queue.enqueueWriteBuffer(input_r,CL_FALSE,0,n*sizeof(T),r);
        queue.enqueueWriteBuffer(input_A,CL_FALSE,0,n*m*sizeof(T),A);
        queue.finish();
        transfer_time=current_time_usecs()-comp_start;
        asm volatile("": : :"memory");

        comp_start=current_time_usecs();
        asm volatile("": : :"memory");
        if(std::is_same<T, float>::value)
        {
            std::vector<cl::Event> wait_events;
            cl::Event e;
            fb.sgemv("sbicg_gemv",FBLAS_NO_TRANSPOSED,n,m,1,input_A,m,input_p,1,0,output_q,1,nullptr, &e);
            wait_events.push_back(e);
            fb.sgemv("sbicg_gemvt",FBLAS_TRANSPOSED,n,m,1,input_A,m,input_r,1,0,output_s,1,nullptr, &e);
            wait_events.push_back(e);
            cl::Event::waitForEvents(wait_events);
        }
        else{
            std::vector<cl::Event> wait_events;
            cl::Event e;
            fb.dgemv("dbicg_gemv",FBLAS_NO_TRANSPOSED,n,m,1,input_A,m,input_p,1,0,output_q,1,nullptr, &e);
            wait_events.push_back(e);
            fb.dgemv("dbicg_gemvt",FBLAS_TRANSPOSED,n,m,1,input_A,m,input_r,1,0,output_s,1,nullptr, &e);
            wait_events.push_back(e);
            cl::Event::waitForEvents(wait_events);

        }
        asm volatile("": : :"memory");
        times.push_back(current_time_usecs()-comp_start);

        asm volatile("": : :"memory");
        comp_start=current_time_usecs();
        queue.enqueueReadBuffer(output_q,CL_FALSE,0,n*sizeof(T),fpga_res_q);
        queue.enqueueReadBuffer(output_s,CL_FALSE,0,m*sizeof(T),fpga_res_s);
        queue.finish();
        transfer_time+=current_time_usecs()-comp_start;
        transfer_times.push_back(transfer_time);

    }


    #if defined (CHECK)
    //check
    if (std::is_same<T, float>::value){
        cblas_sgemv(CblasRowMajor,CblasNoTrans,n,m,1,(float *)A,m,(float *)p,1,0,(float *)q,1);
        cblas_sgemv(CblasRowMajor,CblasTrans,n,m,1,(float *)A,m,(float *)r,1,0,(float *)s,1);
    }
    else{
        cblas_dgemv(CblasRowMajor,CblasNoTrans,n,m,1,(double *)A,m,(double *)p,1,0,(double *)q,1);
        cblas_dgemv(CblasRowMajor,CblasTrans,n,m,1,(double *)A,m,(double *)r,1,0,(double *)s,1);
    }


    bool ok=true;
    bool ok2=true;

    double flteps;
    if (std::is_same<T, float>::value)
        flteps = 1e-4;
    else
        flteps = 1e-6;
    //check

    ok=true;
    for(int i=0;i<n;i++)
    {
        if(!test_equals(((T *)fpga_res_q)[i],((T *)q)[i],flteps))
        {
            ok=false;
        }
    }

    ok2=true;
    for(int i=0;i<m;i++)
    {
        if(!test_equals(((T *)fpga_res_s)[i],((T *)s)[i],flteps))
        {
            ok2=false;
        }
    }
    if(ok && ok2)
        std::cout <<"Result verified." <<std::endl;
    else
        std::cout <<"Error in result." <<std::endl;

        #endif

}



int main(int argc, char *argv[])
{

    //command line argument parsing
    if(argc<13)
    {
        cerr << "Usage: "<< argv[0]<<"  -b <binary_file> -j <json_file> -n <n> -m <m> -r <num runs> -p <precision float/double>"<<endl;
        exit(-1);
    }

    int c;
    int n,m,runs;
    bool double_precision;
    double alpha;
    std::string program_path, json_path;
    while ((c = getopt (argc, argv, "n:b:r:m:s:p:j:")) != -1)
        switch (c)
        {
            case 'n':
                n=atoi(optarg);
                break;
            case 'm':
                m=atoi(optarg);
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
                cerr << "Usage: "<< argv[0]<<""<<endl;
                exit(-1);
        }
    timestamp_t cpu_time;
    std::vector<double> streamed_times,transfer_times;

    //test the streamed vesion
    if(double_precision)
        testNonStreamed<double>(program_path,json_path,n,m,streamed_times,transfer_times,cpu_time,runs);
    else
        testNonStreamed<float>(program_path,json_path,n,m,streamed_times,transfer_times,cpu_time, runs);



    //compute the average and standard deviation of times
    double streamed_mean=0;
    for(auto t:streamed_times)
        streamed_mean+=t;
    streamed_mean/=runs;

    double fblas_stddev,streamed_stddev=0;
    for(auto t:streamed_times)
        streamed_stddev+=((t-streamed_mean)*(t-streamed_mean));
    streamed_stddev=sqrt(streamed_stddev/(runs-1));

    //compute average and standard deviation of transfer times
    double transfer_mean=0, transfer_stddev=0;
    for(auto t:transfer_times)
        transfer_mean+=t;
    transfer_mean/=runs;
    for(auto t:transfer_times)
        transfer_stddev+=((t-transfer_mean)*(t-transfer_mean));
    transfer_stddev=sqrt(transfer_stddev/(runs-1));

    //double fblas_conf_interval_95=1.96*fblas_stddev/sqrt(runs);
    double fblas_conf_interval_99=2.58*fblas_stddev/sqrt(runs);
    //double streamed_conf_interval_95=1.96*streamed_stddev/sqrt(runs);
    double streamed_conf_interval_99=2.58*streamed_stddev/sqrt(runs);
    double transfer_conf_interval_99=2.58*transfer_stddev/sqrt(runs);


    cout << "Computation time over fpga (usecs): "<<streamed_mean<< " (sttdev: " << streamed_stddev<<")"<<endl;
    cout << "Conf interval 99: "<<streamed_conf_interval_99<<endl;
    cout << "Conf interval 99 within " <<(streamed_conf_interval_99/streamed_mean)*100<<"% from mean" <<endl;


}
