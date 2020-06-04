/**
 *  BICG
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

using namespace std;




template <typename T>
void testStreamed(std::string program_path, int n, int m, std::vector<double> &times, std::vector <double> &transfer_times, timestamp_t &cpu_time,int runs)
{

    cl::Platform platform;
    cl::Device device;
    cl::Context context;
    cl::Program program;
    std::vector<std::string> kernel_names;
    int width;
    int tile_size=2048;  //attention, change this if necessary
    if (std::is_same<T, float>::value){
        width = 64;
        cout << "Executing in single precision. " <<endl;
        kernel_names = {"sbicg_gemv","sbicg_gemv_trans", "bicg_read_matrix", "sbicg_read_p", "sbicg_read_q", "sbicg_write_q", "sbicg_read_r", "bicg_read_vector_s_updates", "sbicg_write_s"};
    }
    else{
        width = 32;
        cout << "Executing in double precision. " <<endl;
        kernel_names = {"dbicg_gemv","dbicg_gemv_trans", "bicg_read_matrix", "dbicg_read_p", "dbicg_read_q", "dbicg_write_q", "dbicg_read_r", "bicg_read_vector_s_updates", "dbicg_write_s"};
    }
    std::vector<cl::Kernel> kernels;
    std::vector<cl::CommandQueue> queues;
    std::cout << "Reprogramming device..."<<std::endl;
    IntelFPGAOCLUtils::initEnvironment(platform,device,context,program,program_path,kernel_names,kernels,queues);
    std::cout << "Device reprogrammed."<<std::endl;
    T fpga_res_streamed;
    //copy back the data (data is erased after reprogramming)
    size_t elem_per_module=n*m/4;
    T *A_0,*A_1,*A_2,*A_3;
    void *fpga_res_q, *fpga_res_s;
    void *streamed_res_q, *streamed_res_s;
    void *A,*p,*r,*q,*s;

    posix_memalign ((void **)&A_0, IntelFPGAOCLUtils::AOCL_ALIGNMENT, n*m*sizeof(T)/4);
    posix_memalign ((void **)&A_1, IntelFPGAOCLUtils::AOCL_ALIGNMENT, n*m*sizeof(T)/4);
    posix_memalign ((void **)&A_2, IntelFPGAOCLUtils::AOCL_ALIGNMENT, n*m*sizeof(T)/4);
    posix_memalign ((void **)&A_3, IntelFPGAOCLUtils::AOCL_ALIGNMENT, n*m*sizeof(T)/4);
    posix_memalign ((void **)&streamed_res_q, IntelFPGAOCLUtils::AOCL_ALIGNMENT, n*sizeof(T));
    posix_memalign ((void **)&streamed_res_s, IntelFPGAOCLUtils::AOCL_ALIGNMENT, m*sizeof(T));
    cl::Buffer input_A_0(context, CL_MEM_READ_ONLY|CL_CHANNEL_1_INTELFPGA, elem_per_module*sizeof(T));
    cl::Buffer input_A_1(context, CL_MEM_READ_ONLY|CL_CHANNEL_2_INTELFPGA, elem_per_module*sizeof(T));
    cl::Buffer input_A_2(context, CL_MEM_READ_ONLY|CL_CHANNEL_3_INTELFPGA, elem_per_module*sizeof(T));
    cl::Buffer input_A_3(context, CL_MEM_READ_ONLY|CL_CHANNEL_4_INTELFPGA, elem_per_module*sizeof(T));
    cl::Buffer matrix_A(context, CL_MEM_READ_ONLY|CL_CHANNEL_1_INTELFPGA, n * m* sizeof(T));
    cl::Buffer vector_p(context, CL_MEM_READ_ONLY|CL_CHANNEL_2_INTELFPGA, m * sizeof(T));
    cl::Buffer vector_r(context, CL_MEM_READ_ONLY|CL_CHANNEL_3_INTELFPGA, n * sizeof(T));
    cl::Buffer vector_q(context, CL_MEM_READ_WRITE|CL_CHANNEL_4_INTELFPGA, n * sizeof(T));
    cl::Buffer vector_s(context, CL_MEM_READ_WRITE|CL_CHANNEL_2_INTELFPGA, m * sizeof(T));


     //create data
    posix_memalign ((void **)&A, IntelFPGAOCLUtils::AOCL_ALIGNMENT, n*m*sizeof(T));
    posix_memalign ((void **)&p, IntelFPGAOCLUtils::AOCL_ALIGNMENT, m*sizeof(T));
    posix_memalign ((void **)&q, IntelFPGAOCLUtils::AOCL_ALIGNMENT, n*sizeof(T));
    posix_memalign ((void **)&s, IntelFPGAOCLUtils::AOCL_ALIGNMENT, m*sizeof(T));
    posix_memalign ((void **)&r, IntelFPGAOCLUtils::AOCL_ALIGNMENT, n*sizeof(T));
    generate_vector<T>((T *)p,m);
    generate_vector<T>((T *)r,n);
    generate_matrix<T>((T *)A,n,m);

    //set kernel arguments
    int one=1;
    int zero=0;
    T fzero=0;
    T fone=1;

    std::cout << "Executing streamed version with width: "<<width << "and tile "<<tile_size<<endl;
    int p_repetitions=ceil((T)(n)/tile_size);

    //bicg_gemv
    kernels[0].setArg(0, sizeof(int),&one);
    kernels[0].setArg(1, sizeof(int),&n);
    kernels[0].setArg(2, sizeof(int),&m);
    kernels[0].setArg(3, sizeof(T),&fone);
    kernels[0].setArg(4, sizeof(T),&fzero);

    //bicg_gemvt
    kernels[1].setArg(0, sizeof(int),&one);
    kernels[1].setArg(1, sizeof(int),&n);
    kernels[1].setArg(2, sizeof(int),&m);
    kernels[1].setArg(3, sizeof(T),&fone);
    kernels[1].setArg(4, sizeof(T),&fzero);

    //read_matrix
    kernels[2].setArg(0, sizeof(cl_mem),&input_A_0);
    kernels[2].setArg(1, sizeof(cl_mem),&input_A_1);
    kernels[2].setArg(2, sizeof(cl_mem),&input_A_2);
    kernels[2].setArg(3, sizeof(cl_mem),&input_A_3);
    kernels[2].setArg(4, sizeof(int),&n);
    kernels[2].setArg(5, sizeof(int),&m);
    kernels[2].setArg(6, sizeof(int),&m);

    //read_vector_p
    kernels[3].setArg(0, sizeof(cl_mem),&vector_p);
    kernels[3].setArg(1, sizeof(int),&m);
    kernels[3].setArg(2, sizeof(int),&tile_size);
    kernels[3].setArg(3, sizeof(int),&p_repetitions);

    //read_vector_q
    kernels[4].setArg(0, sizeof(cl_mem),&vector_q);
    kernels[4].setArg(1, sizeof(int),&n);
    kernels[4].setArg(2, sizeof(int),&tile_size);
    kernels[4].setArg(3, sizeof(int),&zero);

    //write_vector_q
    kernels[5].setArg(0, sizeof(cl_mem),&vector_q);
    kernels[5].setArg(1, sizeof(int),&n);
    kernels[5].setArg(2, sizeof(int),&tile_size);

    //read_vector_r
    kernels[6].setArg(0, sizeof(cl_mem),&vector_r);
    kernels[6].setArg(1, sizeof(int),&n);
    kernels[6].setArg(2, sizeof(int),&tile_size);
    kernels[6].setArg(3, sizeof(int),&one);

    //read vector s updateds
    int updated_s=ceil((float)(n)/tile_size);
    kernels[7].setArg(0, sizeof(cl_mem),&vector_s);
    kernels[7].setArg(1, sizeof(int),&m);
    kernels[7].setArg(2, sizeof(int),&updated_s);

    //write vector s
    kernels[8].setArg(0, sizeof(cl_mem),&vector_s);
    kernels[8].setArg(1, sizeof(int),&m);
    kernels[8].setArg(2, sizeof(int),&tile_size);

    //copy the matrix interleaving it into two modules
    size_t offset=0;
    size_t increment=width/4;
    const int loop_it=((int)(m))/width;   //W must be a divisor of M
    /*for(int i=0;i<n;i++)
    {
        for(int j=0;j<loop_it;j++)
        {
            //write to the different memory area
            for(int ii=0;ii<16;ii++)
                A_0[offset+ii]=((T *)A)[i*m+j*width+ii];
            for(int ii=0;ii<16;ii++)
                A_1[offset+ii]=((T *)A)[i*m+j*width+width/4+ii];
            for(int ii=0;ii<16;ii++)
                A_2[offset+ii]=((T *)A)[i*m+j*width+width/2+ii];
            for(int ii=0;ii<16;ii++)
                A_3[offset+ii]=((T *)A)[i*m+j*width+3*width/4+ii];
            offset+=increment;
        }
    }
    */
    for(int i=0;i<n;i++)
    {
        for(int j=0;j<loop_it;j++)
        {
            //write to the different memory area
            for(int ii=0;ii<16;ii++)
                A_0[offset+ii]=((T *)A)[i*m+j*width+ii];
            for(int ii=0;ii<16;ii++)
                A_1[offset+ii]=((T *)A)[i*m+j*width+width/4+ii];
            for(int ii=0;ii<16;ii++)
                A_2[offset+ii]=((T *)A)[i*m+j*width+width/2+ii];
            for(int ii=0;ii<16;ii++)
                A_3[offset+ii]=((T *)A)[i*m+j*width+3*width/4+ii];
            offset+=increment;
        }
    }
    cout << "Ready to go " <<endl;
    //start computation
    for(int i=0;i<runs;i++)
    {

        timestamp_t comp_start=current_time_usecs();


        queues[0].enqueueWriteBuffer(input_A_0, CL_FALSE,0, n*m/4*sizeof(T),A_0);
        queues[0].enqueueWriteBuffer(input_A_1, CL_FALSE,0, n*m/4*sizeof(T),A_1);
        queues[0].enqueueWriteBuffer(input_A_2, CL_FALSE,0, n*m/4*sizeof(T),A_2);
        queues[0].enqueueWriteBuffer(input_A_3, CL_FALSE,0, n*m/4*sizeof(T),A_3);

         //load data to FPGA
        queues[0].enqueueWriteBuffer(vector_p,CL_FALSE,0,m*sizeof(T),p);
        queues[0].enqueueWriteBuffer(vector_r,CL_FALSE,0,n*sizeof(T),r);

        queues[0].finish();
        timestamp_t transf = current_time_usecs() - comp_start;

        comp_start=current_time_usecs();
        asm volatile("": : :"memory");
        for(int i=0;i<kernel_names.size();i++)
            queues[i].enqueueTask(kernels[i]);
        for(int i=0;i<kernel_names.size();i++)
            queues[i].finish();

        asm volatile("": : :"memory");
        times.push_back(current_time_usecs()-comp_start);

        comp_start=current_time_usecs();
        //queues[0].enqueueReadBuffer(vector_q,CL_TRUE,0,n*sizeof(float),streamed_res_q);
        //queues[0].enqueueReadBuffer(vector_s,CL_TRUE,0,m*sizeof(float),streamed_res_s);
        transf += current_time_usecs() - comp_start;
        transfer_times.push_back(transf);
    }

    cout << "Finished" << endl;
    //copy back results
    posix_memalign ((void **)&streamed_res_q, IntelFPGAOCLUtils::AOCL_ALIGNMENT, n*sizeof(T));
    posix_memalign ((void **)&streamed_res_s, IntelFPGAOCLUtils::AOCL_ALIGNMENT, m*sizeof(T));

    queues[0].enqueueReadBuffer(vector_q,CL_TRUE,0,n*sizeof(T),streamed_res_q);
    queues[0].enqueueReadBuffer(vector_s,CL_TRUE,0,m*sizeof(T),streamed_res_s);
    cout << "copied back" << endl;

    //check
    timestamp_t comp_start=current_time_usecs();
    if (std::is_same<T, float>::value){
        cblas_sgemv(CblasRowMajor,CblasNoTrans,n,m,1,(float *)A,m,(float *)p,1,0,(float *)q,1);
        cblas_sgemv(CblasRowMajor,CblasTrans,n,m,1,(float *)A,m,(float *)r,1,0,(float *)s,1);
    }
    else{
        cblas_dgemv(CblasRowMajor,CblasNoTrans,n,m,1,(double *)A,m,(double *)p,1,0,(double *)q,1);
        cblas_dgemv(CblasRowMajor,CblasTrans,n,m,1,(double *)A,m,(double *)r,1,0,(double *)s,1);
    }

        cpu_time=current_time_usecs()-comp_start;

    //forse meglio passare a norma
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
        if(!test_equals(((T *)streamed_res_q)[i],((T *)q)[i],flteps))
        {
            ok=false;
            std::cout << "Error q: " << ((T *)streamed_res_q)[i] << " != " << ((T *)q)[i] <<std::endl;
        }
    }

    ok2=true;
    for(int i=0;i<m;i++)
    {
        if(!test_equals(((T *)streamed_res_s)[i],((T *)s)[i],flteps))
        {
            ok2=false;
            std::cout << "Error s: " << ((T *)streamed_res_s)[i] << " != " << ((T *)s)[i] <<std::endl;
        }
    }
    if(ok && ok2)
        std::cout <<"Result verified." <<std::endl;
    else
        std::cout <<"Error in result." <<std::endl;

}



int main(int argc, char *argv[])
{

    //command line argument parsing
    if(argc<11)
    {
        cerr << "Usage: "<< argv[0]<<"  -b <binary_streamed> -n <n> -m <m> -r <num runs> -p <precision float/double>"<<endl;
        exit(-1);
    }

    int c;
    int n,m,runs;
    bool double_precision;
    double alpha;
    std::string program_path;
    while ((c = getopt (argc, argv, "n:b:r:m:s:p:")) != -1)
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
        testStreamed<double>(program_path,n,m,streamed_times,transfer_times,cpu_time,runs);
    else
        testStreamed<float>(program_path,n,m,streamed_times,transfer_times,cpu_time, runs);



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

    cout << "Computation time over cpu (usecs): "<<cpu_time<<endl;


    cout << "Computation time over fpga with streamed (usecs): "<<streamed_mean<< " (sttdev: " << streamed_stddev<<")"<<endl;
    cout << "Streamed Conf interval 99: "<<streamed_conf_interval_99<<endl;
    cout << "Streamed Conf interval 99 within " <<(streamed_conf_interval_99/streamed_mean)*100<<"% from mean" <<endl;


}
