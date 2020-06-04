/**
 * Matrix matrix multiplication test for fully unrolled routines
 *
 * Attention: being fully unrolled, the routine works with the matrix sizes given during compilation
 */
#include <stdio.h>
#include <string>
#include <iostream>
#include <fstream>
#include <mkl.h>
#include <math.h>
#include "../../include/utils/ocl_utils.hpp"
#include "../../include/utils/utils.hpp"
#include "../../include/utils/test.hpp"
#include "../../include/utils/data_generators.hpp"


#define CHECK //disable to avoid checking the result

using namespace std;

template <typename T>
void evaluate(std::string program_path, T alpha, T beta, int num_gemm, std::vector<double> &times, timestamp_t &cpu_time, int runs){

    T *A,*B,*fpgaC;
    T *C;
    int n,m,k;
    if (std::is_same<T, float>::value){
        cout << "Single precision: working with 4x4 matrices" <<endl;
        n=4;
        m=4;
        k=4;
    }
    else{
        cout << "Double precision: working with 4x4 matrices" <<endl;
        n=4;
        m=4;
        k=4;
    }

    posix_memalign ((void **)&A, IntelFPGAOCLUtils::AOCL_ALIGNMENT, num_gemm*n*k*sizeof(T));
    posix_memalign ((void **)&B, IntelFPGAOCLUtils::AOCL_ALIGNMENT, num_gemm*k*m*sizeof(T));
    posix_memalign ((void **)&C, IntelFPGAOCLUtils::AOCL_ALIGNMENT, num_gemm*n*m*sizeof(T));
    posix_memalign ((void **)&fpgaC, IntelFPGAOCLUtils::AOCL_ALIGNMENT, num_gemm*n*m*sizeof(T));
    for(int i=0;i<num_gemm;i++)
    {

        if (std::is_same<T, float>::value){
            generate_float_matrix((float*)(A+i*(n*k)),n,k);
            generate_float_matrix((float*)(B+i*(m*k)),k,m);
            generate_float_matrix((float*)(C+i*(n*m)),n,m);
        }
        else{
            generate_double_matrix((double*)(A+i*(n*k)),n,k);
            generate_double_matrix((double*)(B+i*(m*k)),k,m);
            generate_double_matrix((double*)(C+i*(n*m)),n,m);
        }
    }



    cl_int status;
    cl::Platform  platform;
    cl::Device device;
    cl::Context context;
    cl::Program program;
    std::vector<cl::Kernel> kernels;
    std::vector<cl::CommandQueue> queues;
    std::vector<std::string> kernel_names={"sgemm", "read_matrix_A", "read_matrix_B", "read_matrix_C", "write_matrix_C"};
    IntelFPGAOCLUtils::initEnvironment(platform,device,context,program,program_path,kernel_names, kernels,queues);

    //create memory buffers
    cl::Buffer input_A(context,CL_MEM_READ_ONLY|CL_CHANNEL_1_INTELFPGA,num_gemm*n*k*sizeof(T));
    cl::Buffer input_B(context,CL_MEM_READ_ONLY|CL_CHANNEL_2_INTELFPGA,num_gemm*k*m*sizeof(T));
    cl::Buffer output_C(context,CL_MEM_READ_WRITE|CL_CHANNEL_3_INTELFPGA,num_gemm*n*m*sizeof(T));

    queues[0].enqueueWriteBuffer(input_A,CL_TRUE,0,num_gemm*n*k*sizeof(T),A);
    queues[0].enqueueWriteBuffer(input_B,CL_TRUE,0,num_gemm*k*m*sizeof(T),B);

    //sgemm
    kernels[0].setArg(0,sizeof(int),&n);
    kernels[0].setArg(1,sizeof(int),&m);
    kernels[0].setArg(2,sizeof(int),&k);
    kernels[0].setArg(3,sizeof(T),&alpha);
    kernels[0].setArg(4,sizeof(T),&beta);
    kernels[0].setArg(5,sizeof(int),&num_gemm);

    //read_A
    kernels[1].setArg(0,sizeof(cl_mem),&input_A);
    kernels[1].setArg(1,sizeof(int),&num_gemm);

    //read_B
    kernels[2].setArg(0,sizeof(cl_mem),&input_B);
    kernels[2].setArg(1,sizeof(int),&num_gemm);

    //read_C
    kernels[3].setArg(0,sizeof(cl_mem),&output_C);
    kernels[3].setArg(1,sizeof(int),&num_gemm);

    //write_C
    kernels[4].setArg(0,sizeof(cl_mem),&output_C);
    kernels[4].setArg(1,sizeof(int),&num_gemm);


    for(int r=0;r<runs;r++)
    {
        queues[0].enqueueWriteBuffer(output_C,CL_TRUE,0,num_gemm*n*m*sizeof(T),C);

        timestamp_t fpga_start = current_time_usecs();
        for(int i=0;i<kernels.size();i++)
            queues[i].enqueueTask(kernels[i]);

        //wait
        for(int i=0;i<kernels.size();i++)
            queues[i].finish();

        times.push_back((double)((current_time_usecs()-fpga_start)));
        if(r==0)
            queues[0].enqueueReadBuffer(output_C,CL_TRUE,0,num_gemm*n*m*sizeof(T),fpgaC);
    }


    bool ok = true;

    mkl_set_dynamic(0);
    mkl_set_num_threads_local(10);

    //batched gemm
    #define    GRP_COUNT    1

    MKL_INT    m_mkl[GRP_COUNT] = {m};
    MKL_INT    k_mkl[GRP_COUNT] = {k};
    MKL_INT    n_mkl[GRP_COUNT] = {n};

    MKL_INT    lda_mkl[GRP_COUNT] = {k};
    MKL_INT    ldb_mkl[GRP_COUNT] = {m};
    MKL_INT    ldc_mkl[GRP_COUNT] = {m};

    CBLAS_TRANSPOSE    transA[GRP_COUNT] = {CblasNoTrans};
    CBLAS_TRANSPOSE    transB[GRP_COUNT] = {CblasNoTrans};

    T    alpha_mkl[GRP_COUNT] = {(T)alpha};
    T    beta_mkl[GRP_COUNT] = {(T)beta};

    T   *a_array[num_gemm];
    T   *b_array[num_gemm];
    T   *c_array[num_gemm];
    for(int i=0;i<num_gemm;i++){
        a_array[i]=&A[i*n*m];
        b_array[i]=&B[i*n*m];
        c_array[i]=&C[i*n*m];
    }


    MKL_INT    size_per_grp[GRP_COUNT] = {num_gemm};

    timestamp_t start_time = current_time_usecs();

    // Call cblas_dgemm_batch
    if (std::is_same<T, float>::value){
        cblas_sgemm_batch (
                CblasRowMajor,
                transA,
                transB,
                n_mkl,
                m_mkl,
                k_mkl,
                (float *)alpha_mkl,
                (const float **)a_array,
                lda_mkl,
                (const float **)b_array,
                ldb_mkl,
                (float *)beta_mkl,
                (float **)c_array,
                ldc_mkl,
                GRP_COUNT,
                size_per_grp);
    }
    else{
        cblas_dgemm_batch (
            CblasRowMajor,
            transA,
            transB,
            n_mkl,
            m_mkl,
            k_mkl,
            (double *)alpha_mkl,
            (const double **)a_array,
            lda_mkl,
            (const double **)b_array,
            ldb_mkl,
            (double *)beta_mkl,
            (double **)c_array,
            ldc_mkl,
            GRP_COUNT,
            size_per_grp);
    }

    cpu_time = current_time_usecs() - start_time;
    for(int i=0;i<num_gemm;i++){
        const double flteps = 1e-4;
        bool res=compare_matrices(fpgaC+i*(n*m),C+i*(n*m),n,m,flteps);
        ok &=res;
    }


    if(ok) std::cout << "Result verified!" <<std::endl;
    else std::cout << "Error!!!!!!!" << std::endl;


}
int main(int argc, char *argv[])
{

    //command line argument parsing
    if(argc<11)
    {
        cerr << "Fully unrolled GEMM " <<endl;
        cerr << "Usage: "<< argv[0]<<" -b <binary file> -a <alpha> -c <beta> -l <num_gemms> -r <runs> -p <precision float/double>"<<endl;
        exit(-1);
    }
    int c;
    int n,m,k;
    int num_gemm;
    bool double_precision;
    std::string program_path;
    int runs = 10;
    float alpha, beta;
    while ((c = getopt (argc, argv, "b:a:c:l:p:r:")) != -1)
        switch (c)
        {
            case 'b':
                program_path=std::string(optarg);
                break;
            case 'a':
                alpha=atof(optarg);
                break;
            case 'c':
                beta=atof(optarg);
                break;
             case 'r':
                runs=atoi(optarg);
                break;
            case 'l':
                num_gemm = atoi(optarg);
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
                cerr << "Usage: "<< argv[0]<<" -b <binary file> -a <alpha> -c <beta> -l <num_gemms> -p <precision float/double>"<<endl;
                exit(-1);
        }


    //init data: matrices should be memorized as flat arrays (useful for kernel execution)
    std::vector<double> times;
    timestamp_t cpu_time;
    if(double_precision)
        evaluate<double>(program_path,alpha, beta, num_gemm, times, cpu_time, runs );
    else
        evaluate<float>(program_path,alpha, beta, num_gemm, times, cpu_time, runs );

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

    double computation_gops=num_gemm*((double)(m)*n*(2*k-1)+m*n)/1000000000;
    double measured_gops=computation_gops/((mean)/1000000.0);
    cout << std::fixed;
    cout << "FPGA Computation time (usec): " << mean << " (sttdev: " << stddev<<")"<<endl;
    std::cout << "CPU Computation time: " << cpu_time << " usecs"<<std::endl;



}
