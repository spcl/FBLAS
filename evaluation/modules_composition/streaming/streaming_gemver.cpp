/**
    Host program for gemver implemented using streaming modules
     * GEMVER
     * B=A+u1v1**T+u2v2**T
     * x=beta*B**Ty+z
     * w=alpha*Bx
     *
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <string>
#include <cblas.h>
#include <sstream>

#include "../../../include/utils/ocl_utils.hpp"
#include "../../../include/utils/utils.hpp"
#include "../../../include/utils/test.hpp"
#include "../../../include/utils/data_generators.hpp"

using namespace std;



template <typename T>
void testStreaming(std::string program_path, int n, T alpha, T beta, std::vector<double> &times, std::vector <double> &transfer_times, timestamp_t &cpu_time, int runs)
{

    //Attention: these must be the same of the codegenerated routines
    int width;
    const int tile_size=2048;

    cl::Platform platform;
    cl::Device device;
    cl::Context context;
    cl::Program program;
    std::vector<std::string> kernel_names;
    if (std::is_same<T, float>::value){
        width = 32;
        cout << "Executing in single precision. " <<endl;
        kernel_names = {"sgemver_ger1","sgemver_read_u1", "sgemver_read_v1", "sgemver_read_matrix_A", "sgemver_ger2",\
        "sgemver_read_u2", "sgemver_read_v2", "write_and_forward_matrix_B", "sgemver_gemv_trans", "sgemver_read_y", \
        "sgemver_read_z", "sgemver_write_x","sgemver_gemv","sgemver_read_x","sgemver_read_w","sgemver_read_matrix_B","sgemver_write_w"};

    }
    else{
        width = 16;
        cout << "Executing in double precision. " <<endl;
        kernel_names = {"dgemver_ger1","dgemver_read_u1", "dgemver_read_v1", "dgemver_read_matrix_A", "dgemver_ger2",\
        "dgemver_read_u2", "dgemver_read_v2", "write_and_forward_matrix_B", "dgemver_gemv_trans", "dgemver_read_y", \
        "dgemver_read_z", "dgemver_write_x","dgemver_gemv","dgemver_read_x","dgemver_read_w","dgemver_read_matrix_B","dgemver_write_w"};
    }

    std::vector<cl::Kernel> kernels;
    std::vector<cl::CommandQueue> queues;
    const int num_kernels=kernel_names.size();
    cout << "Reprogramming device ..." <<endl;
    IntelFPGAOCLUtils::initOpenCL(platform,device,context,program,program_path);
    IntelFPGAOCLUtils::createCommandQueues(context,device,queues,num_kernels);
    IntelFPGAOCLUtils::createKernels(program,kernel_names,kernels);
    cout << "Device reprogrammed." <<endl;
    float fpga_res_streamed;


    //create data
    void *A,*B,*u1,*u2,*w,*v1,*v2,*z,*y,*x;
    int ret;
    ret = posix_memalign ((void **)&A, IntelFPGAOCLUtils::AOCL_ALIGNMENT, n*n*sizeof(T));
    ret = posix_memalign ((void **)&B, IntelFPGAOCLUtils::AOCL_ALIGNMENT, n*n*sizeof(T));
    ret = posix_memalign ((void **)&u1, IntelFPGAOCLUtils::AOCL_ALIGNMENT, n*sizeof(T));
    ret = posix_memalign ((void **)&u2, IntelFPGAOCLUtils::AOCL_ALIGNMENT, n*sizeof(T));
    ret = posix_memalign ((void **)&w, IntelFPGAOCLUtils::AOCL_ALIGNMENT, n*sizeof(T));
    ret = posix_memalign ((void **)&v1, IntelFPGAOCLUtils::AOCL_ALIGNMENT, n*sizeof(T));
    ret = posix_memalign ((void **)&v2, IntelFPGAOCLUtils::AOCL_ALIGNMENT, n*sizeof(T));
    ret = posix_memalign ((void **)&z, IntelFPGAOCLUtils::AOCL_ALIGNMENT, n*sizeof(T));
    ret = posix_memalign ((void **)&y, IntelFPGAOCLUtils::AOCL_ALIGNMENT, n*sizeof(T));
    ret = posix_memalign ((void **)&x, IntelFPGAOCLUtils::AOCL_ALIGNMENT, n*sizeof(T));
    generate_vector<T>((T *)u1,n);
    generate_vector<T>((T *)u2,n);
    generate_vector<T>((T *)y,n);
    generate_vector<T>((T *)z,n);
    generate_vector<T>((T *)v1,n);
    generate_vector<T>((T *)v2,n);
    generate_matrix<T>((T *)A,n,n);


    //copy back the data (data is erased after reprogramming)
    T *fpga_res, *res_fpga_B;
    posix_memalign ((void **)&fpga_res, IntelFPGAOCLUtils::AOCL_ALIGNMENT, n*sizeof(T));
    posix_memalign ((void **)&res_fpga_B, IntelFPGAOCLUtils::AOCL_ALIGNMENT, n*n*sizeof(T));
    cl::Buffer fpga_A(context, CL_MEM_READ_ONLY|CL_CHANNEL_1_INTELFPGA, n *n * sizeof(T));
    cl::Buffer fpga_B(context, CL_MEM_READ_WRITE|CL_CHANNEL_2_INTELFPGA, n *n * sizeof(T));
    cl::Buffer fpga_u1(context, CL_MEM_READ_ONLY|CL_CHANNEL_3_INTELFPGA, n *sizeof(T));
    cl::Buffer fpga_u2(context, CL_MEM_READ_ONLY|CL_CHANNEL_3_INTELFPGA, n *sizeof(T));
    cl::Buffer fpga_v1(context, CL_MEM_READ_ONLY|CL_CHANNEL_4_INTELFPGA, n * sizeof(T));
    cl::Buffer fpga_v2(context, CL_MEM_READ_ONLY|CL_CHANNEL_4_INTELFPGA, n * sizeof(T));
    cl::Buffer fpga_y(context, CL_MEM_READ_ONLY|CL_CHANNEL_1_INTELFPGA, n * sizeof(T));
    cl::Buffer fpga_x(context, CL_MEM_READ_WRITE|CL_CHANNEL_3_INTELFPGA, n * sizeof(T));
    cl::Buffer fpga_z(context, CL_MEM_READ_ONLY|CL_CHANNEL_4_INTELFPGA, n * sizeof(T));
    cl::Buffer fpga_w(context, CL_MEM_WRITE_ONLY|CL_CHANNEL_4_INTELFPGA, n * sizeof(T));


    int one=1;

    cout << "Streamed version executed with width: " << width << " and tile size "<<tile_size<<endl;
    int u_repetitions=ceil((float)(n)/tile_size); //this is necessary for ger routines
    int x_repetitions=ceil((float)(n)/tile_size); //this is necessary for ger routines
    T fone=1.0f;

    //set args

    //sger1
    kernels[0].setArg(0,sizeof(T),&fone);
    kernels[0].setArg(1,sizeof(int),&n);
    kernels[0].setArg(2,sizeof(int),&n);

    //read vectors u1,v1 and matrix A
    kernels[1].setArg(0,sizeof(cl_mem),&fpga_u1);
    kernels[1].setArg(1,sizeof(unsigned int),&n);
    kernels[1].setArg(2,sizeof(unsigned int),&tile_size);
    kernels[1].setArg(3,sizeof(unsigned int),&u_repetitions);
    kernels[2].setArg(0,sizeof(cl_mem),&fpga_v1);
    kernels[2].setArg(1,sizeof(unsigned int),&n);
    kernels[2].setArg(2,sizeof(unsigned int),&tile_size);
    kernels[2].setArg(3,sizeof(unsigned int),&one);
    kernels[3].setArg(0,sizeof(cl_mem),&fpga_A);
    kernels[3].setArg(1,sizeof(unsigned int),&n);
    kernels[3].setArg(2,sizeof(unsigned int),&n);
    kernels[3].setArg(3,sizeof(unsigned int),&n);

    //sger 2
    kernels[4].setArg(0,sizeof(T),&fone);
    kernels[4].setArg(1,sizeof(int),&n);
    kernels[4].setArg(2,sizeof(int),&n);
    //read vectors u2,v2 and write matrix B
    kernels[5].setArg(0,sizeof(cl_mem),&fpga_u2);
    kernels[5].setArg(1,sizeof(unsigned int),&n);
    kernels[5].setArg(2,sizeof(unsigned int),&tile_size);
    kernels[5].setArg(3,sizeof(unsigned int),&u_repetitions);
    kernels[6].setArg(0,sizeof(cl_mem),&fpga_v2);
    kernels[6].setArg(1,sizeof(unsigned int),&n);
    kernels[6].setArg(2,sizeof(unsigned int),&tile_size);
    kernels[6].setArg(3,sizeof(unsigned int),&one);
    kernels[7].setArg(0,sizeof(cl_mem),&fpga_B);
    kernels[7].setArg(1,sizeof(unsigned int),&n);
    kernels[7].setArg(2,sizeof(unsigned int),&n);
    kernels[7].setArg(3,sizeof(unsigned int),&n);

    int y_repetitions=ceil((float)(n)/tile_size);


    //gemv trans
    kernels[8].setArg(0,sizeof(int),&one);
    kernels[8].setArg(1,sizeof(int),&n);
    kernels[8].setArg(2,sizeof(int),&n);
    kernels[8].setArg(3,sizeof(T),&beta);
    kernels[8].setArg(4,sizeof(T),&fone);

    //read vectors y and z and matrix b trans
    kernels[9].setArg(0,sizeof(cl_mem),&fpga_y);
    kernels[9].setArg(1,sizeof(unsigned int),&n);
    kernels[9].setArg(2,sizeof(unsigned int),&tile_size);
    kernels[9].setArg(3,sizeof(unsigned int),&y_repetitions);
    kernels[10].setArg(0,sizeof(cl_mem),&fpga_z);
    kernels[10].setArg(1,sizeof(unsigned int),&n);
    kernels[10].setArg(2,sizeof(unsigned int),&tile_size);
    kernels[10].setArg(3,sizeof(unsigned int),&one);
    kernels[11].setArg(0,sizeof(cl_mem),&fpga_x);
    kernels[11].setArg(1,sizeof(unsigned int),&n);
    kernels[11].setArg(2,sizeof(unsigned int),&tile_size);

    //gemv
    int zero=0;
    T fzero=0;
    kernels[12].setArg(0,sizeof(int),&one);
    kernels[12].setArg(1,sizeof(int),&n);
    kernels[12].setArg(2,sizeof(int),&n);
    kernels[12].setArg(3,sizeof(T),&alpha);
    kernels[12].setArg(4,sizeof(T),&fzero);

    //readx, read w (dummy), matrix B
    kernels[13].setArg(0,sizeof(cl_mem),&fpga_x);
    kernels[13].setArg(1,sizeof(unsigned int),&n);
    kernels[13].setArg(2,sizeof(unsigned int),&tile_size);
    kernels[13].setArg(3,sizeof(unsigned int),&x_repetitions);

    kernels[14].setArg(0,sizeof(cl_mem),&fpga_w);
    kernels[14].setArg(1,sizeof(unsigned int),&n);
    kernels[14].setArg(2,sizeof(unsigned int),&tile_size);
    kernels[14].setArg(3,sizeof(unsigned int),&zero);

    kernels[15].setArg(0,sizeof(cl_mem),&fpga_B);
    kernels[15].setArg(1,sizeof(unsigned int),&n);
    kernels[15].setArg(2,sizeof(unsigned int),&n);
    kernels[15].setArg(3,sizeof(unsigned int),&n);

    //write w
    kernels[16].setArg(0,sizeof(cl_mem),&fpga_w);
    kernels[16].setArg(1,sizeof(unsigned int),&n);
    kernels[16].setArg(2,sizeof(unsigned int),&tile_size);


    //
    timestamp_t comp_start;
    timestamp_t transfer_time=0;
    for(int i=0;i<runs;i++)
    {
        comp_start=current_time_usecs();
        queues[0].enqueueWriteBuffer(fpga_u1,CL_FALSE,0,n*sizeof(T),u1);
        queues[0].enqueueWriteBuffer(fpga_u2,CL_FALSE,0,n*sizeof(T),u2);
        queues[0].enqueueWriteBuffer(fpga_v1,CL_FALSE,0,n*sizeof(T),v1);
        queues[0].enqueueWriteBuffer(fpga_v2,CL_FALSE,0,n*sizeof(T),v2);
        queues[0].enqueueWriteBuffer(fpga_A,CL_FALSE,0,n*n*sizeof(T),A);
        queues[0].enqueueWriteBuffer(fpga_y,CL_FALSE,0,n*sizeof(T),y);
        queues[0].enqueueWriteBuffer(fpga_z,CL_FALSE,0,n*sizeof(T),z);
        queues[0].finish();
        transfer_time=current_time_usecs()-comp_start;

        timestamp_t comp_start=current_time_usecs();
        asm volatile("": : :"memory");
        //first of all we start ger and the gemv and wait for them to finish,
        //then we will start the two gemv
        //we use opencl events

       // std::vector<cl::Event> wait_events;
        for(int i=0;i<12;i++)
            queues[i].enqueueTask(kernels[i]);
        for(int i=0;i<12;i++)
            queues[i].finish();

        for(int i=12;i<num_kernels;i++)
            queues[i].enqueueTask(kernels[i]);


        //wait
        for(int i=12;i<num_kernels;i++)
            queues[i].finish();
        asm volatile("": : :"memory");
        times.push_back(current_time_usecs()-comp_start);

        comp_start=current_time_usecs();
        queues[0].enqueueReadBuffer(fpga_w,CL_TRUE,0,n*sizeof(T),fpga_res);
        queues[0].enqueueReadBuffer(fpga_B,CL_TRUE,0,n*n*sizeof(T),res_fpga_B);
        transfer_time+=current_time_usecs()-comp_start;
        transfer_times.push_back(transfer_time);
    }

    //check

     //check
    comp_start=current_time_usecs();
    if (std::is_same<T, float>::value){
        cblas_scopy(n*n,(float *)A,1,(float *)B,1);
        cblas_sger(CblasRowMajor,n,n,1,(float *)u1,1,(float *)v1,1,(float *)B,n);
        cblas_sger(CblasRowMajor,n,n,1,(float *)u2,1,(float *)v2,1,(float *)B,n);
        cblas_scopy(n,(float *)z,1,(float *)x,1);
        cblas_sgemv(CblasRowMajor,CblasTrans,n,n,beta,(float *)B,n,(float *)y,1,1,(float *)x,1);
        cblas_sgemv(CblasRowMajor,CblasNoTrans,n,n,alpha,(float *)B,n,(float *)x,1,0,(float *)w,1);
    }
    else{
        cblas_dcopy(n*n,(double *)A,1,(double *)B,1);
        cblas_dger(CblasRowMajor,n,n,1,(double *)u1,1,(double *)v1,1,(double *)B,n);
        cblas_dger(CblasRowMajor,n,n,1,(double *)u2,1,(double *)v2,1,(double *)B,n);
        cblas_dcopy(n,(double *)z,1,(double *)x,1);
        cblas_dgemv(CblasRowMajor,CblasTrans,n,n,beta,(double *)B,n,(double *)y,1,1,(double *)x,1);
        cblas_dgemv(CblasRowMajor,CblasNoTrans,n,n,alpha,(double *)B,n,(double *)x,1,0,(double *)w,1);
    }
    cpu_time=current_time_usecs()-comp_start;
    bool result_verified = true;


    if (std::is_same<T, float>::value){
        double nrminf=0, nrminf_o=0, error;


        for(int i=0;i<n;i++){
            nrminf+=fabs(fpga_res[i]-((float *)w)[i]);
            nrminf_o+=fabs(((float *)w)[i]);

        }
        if(nrminf==0 && nrminf_o ==0)
            error=0;
        else
            error=nrminf/nrminf_o;
        result_verified = error < 1e-4;


    }
    else{

         double nrminf=0, nrminf_o=0, error;
          for(int i=0;i<n;i++){
            nrminf+=fabs(fpga_res[i]-((double *)w)[i]);
            nrminf_o+=fabs(((double *)w)[i]);
            //result_verified &= test_equals(fpga_res[i],((float *)w)[i],1e-4);

        }
        if(nrminf==0 && nrminf_o ==0)
            error=0;
        else
            error=nrminf/nrminf_o;
        result_verified = error < 1e-6;

    }

    if(result_verified)
        cout << "Result verified. " <<endl;
    else
        cout << "Error, result not verified" <<endl;


}

int main(int argc, char *argv[])
{

    //command line argument parsing
    if(argc<13)
    {
        cerr << "Usage: "<< argv[0]<<" -b <binary file> -n <length of the vectors> -a <alpha> -c <beta>  -r <num runs> -p <precision float/double>"<<endl;
        exit(-1);
    }

    int c;
    int n,runs;
    bool double_precision;
    double alpha, beta;
    std::string program_path;
    std::string json_path;
    while ((c = getopt (argc, argv, "n:b:r:a:p:c:")) != -1)
        switch (c)
        {
            case 'n':
                n=atoi(optarg);
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
                cerr << "Usage: "<< argv[0]<<" -b <binary file> -n <length of the vectors> -a <alpha> -c <beta>  -r <num runs> -p <precision float/double>"<<endl;
                exit(-1);
        }
    timestamp_t comp_start, cpu_time;
    std::vector<double> streamed_times,transfer_times;
    if(double_precision)
        testStreaming<double>(program_path,n,alpha,beta,streamed_times, transfer_times,cpu_time,runs);
    else
        testStreaming<float>(program_path,n,alpha,beta,streamed_times, transfer_times,cpu_time,runs);

    //compute the average and standard deviation of times
    double streamed_mean=0;
    for(auto t:streamed_times)
        streamed_mean+=t;
    streamed_mean/=runs;

    double streamed_stddev=0;
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

    double streamed_conf_interval_99=2.58*streamed_stddev/sqrt(runs);
    double transfer_conf_interval_99=2.58*transfer_stddev/sqrt(runs);


    cout << "Computation time over cpu (usecs): "<<cpu_time<<endl;
    cout << "Transfer time mesured with FBLAS (usec) " << transfer_mean << " (sttdev: " << transfer_stddev<<")"<<endl;
    cout << "Computation time over fpga with streamed (usecs): "<<streamed_mean<< " (sttdev: " << streamed_stddev<<")"<<endl;
    cout << "Streamed Conf interval 99: "<<streamed_conf_interval_99<<endl;
    cout << "Streamed Conf interval 99 within " <<(streamed_conf_interval_99/streamed_mean)*100<<"% from mean" <<endl;

}
