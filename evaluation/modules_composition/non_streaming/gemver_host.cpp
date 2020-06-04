/**
    Host program for gemver implemented using hosst api
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
#include "../../../include/fblas_environment.hpp"

using namespace std;



template <typename T>
void testNonStreaming(std::string program_path, std::string json_path, int n, T alpha, T beta, std::vector<double> &times, std::vector <double> &transfer_times, timestamp_t &cpu_time, int runs)
{

    FBLASEnvironment fb(program_path,json_path);

    cl::CommandQueue queue;
    cl::Context context=fb.get_context();
    cl::Device device=fb.get_device();
    IntelFPGAOCLUtils::createCommandQueue(context,device,queue);

    timestamp_t comp_start;
    //create data
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
    T *fpga_res;
    posix_memalign ((void **)&fpga_res, IntelFPGAOCLUtils::AOCL_ALIGNMENT, n*sizeof(T));
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

    if(std::is_same<T, float>::value)
        cout << "Single precision version. " <<endl;
    else
        cout << "Double precision version. " <<endl;

    timestamp_t transfer_time=0;
    for(int i=0;i<runs;i++)
    {
        comp_start=current_time_usecs();
        queue.enqueueWriteBuffer(fpga_u1,CL_FALSE,0,n*sizeof(T),u1);
        queue.enqueueWriteBuffer(fpga_u2,CL_FALSE,0,n*sizeof(T),u2);
        queue.enqueueWriteBuffer(fpga_v1,CL_FALSE,0,n*sizeof(T),v1);
        queue.enqueueWriteBuffer(fpga_v2,CL_FALSE,0,n*sizeof(T),v2);
        queue.enqueueWriteBuffer(fpga_A,CL_FALSE,0,n*n*sizeof(T),A);
        queue.enqueueWriteBuffer(fpga_y,CL_FALSE,0,n*sizeof(T),y);
        queue.enqueueWriteBuffer(fpga_z,CL_FALSE,0,n*sizeof(T),z);
        queue.finish();
        transfer_time=current_time_usecs()-comp_start;

        comp_start=current_time_usecs();
        asm volatile("": : :"memory");
        if(std::is_same<T, float>::value){
            fb.scopy("sgemver_copy",n*n,fpga_A,1,fpga_B,1);
            fb.sger("sgemver_ger",n,n,1,fpga_u1,1,fpga_v1,1,fpga_B,n);
            fb.sger("sgemver_ger",n,n,1,fpga_u2,1,fpga_v2,1,fpga_B,n);
            fb.scopy("sgemver_copy",n,fpga_z,1,fpga_x,1);
            fb.sgemv("sgemver_gemv_trans",FBLAS_TRANSPOSED,n,n,beta,fpga_B,n,fpga_y,1,1,fpga_x,1);
            fb.sgemv("sgemver_gemv",FBLAS_NO_TRANSPOSED,n,n,alpha,fpga_B,n,fpga_x,1,0,fpga_w,1);
        }
        else{
            fb.dcopy("dgemver_copy",n*n,fpga_A,1,fpga_B,1);
            fb.dger("dgemver_ger",n,n,1,fpga_u1,1,fpga_v1,1,fpga_B,n);
            fb.dger("dgemver_ger",n,n,1,fpga_u2,1,fpga_v2,1,fpga_B,n);
            fb.dcopy("dgemver_copy",n,fpga_z,1,fpga_x,1);
            fb.dgemv("dgemver_gemv_trans",FBLAS_TRANSPOSED,n,n,beta,fpga_B,n,fpga_y,1,1,fpga_x,1);
            fb.dgemv("dgemver_gemv",FBLAS_NO_TRANSPOSED,n,n,alpha,fpga_B,n,fpga_x,1,0,fpga_w,1);

        }
        asm volatile("": : :"memory");
        times.push_back(current_time_usecs()-comp_start);

        comp_start=current_time_usecs();
        queue.enqueueReadBuffer(fpga_w,CL_TRUE,0,n*sizeof(T),fpga_res);
        transfer_time+=current_time_usecs()-comp_start;
        transfer_times.push_back(transfer_time);

    }


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
        if(error > 1e-4)
             cout << "Error " << error<< endl;

    }
    else{

         double nrminf=0, nrminf_o=0, error;
          for(int i=0;i<n;i++){
            cout << fpga_res[i] << " " <<  ((double *)w)[i] <<endl;
            nrminf+=fabs(fpga_res[i]-((double *)w)[i]);
            nrminf_o+=fabs(((double *)w)[i]);

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
    if(argc<15)
    {
        cerr << "Usage: "<< argv[0]<<" -b <binary file> -j <json_file> -n <length of the vectors> -a <alpha> -c <beta>  -r <num runs> -p <precision float/double>"<<endl;
        exit(-1);
    }

    int c;
    int n,runs;
    bool double_precision;
    double alpha, beta;
    std::string program_path;
    std::string json_path;
    while ((c = getopt (argc, argv, "n:b:r:a:p:c:j:")) != -1)
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
                cerr << "Usage: "<< argv[0]<<" -b <binary file> -n <length of the vectors> -a <alpha> -c <beta>  -r <num runs> -p <precision float/double>"<<endl;
                exit(-1);
        }
    timestamp_t comp_start, cpu_time;
    std::vector<double> streamed_times,transfer_times;
    if(double_precision)
        testNonStreaming<double>(program_path,json_path, n,alpha,beta,streamed_times, transfer_times,cpu_time,runs);
    else
        testNonStreaming<float>(program_path, json_path, n,alpha,beta,streamed_times, transfer_times,cpu_time,runs);

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


    cout << "Computation time over FPGA (usecs): "<<streamed_mean<< " (sttdev: " << streamed_stddev<<")"<<endl;
    cout << "Conf interval 99: "<<streamed_conf_interval_99<<endl;
    cout << "Conf interval 99 within " <<(streamed_conf_interval_99/streamed_mean)*100<<"% from mean" <<endl;

}
