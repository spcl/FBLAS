/*
  Test for  GEMM product: can be single and double precision

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
#include "../../include/utils/data_generators.hpp"

#include "../../include/fblas_environment.hpp"


using namespace std;
#define CHECK //enable to perform check


void singlePrecision(std::string program_path, std::string json_path, int n, int k, int m, timestamp_t &fpga_time, float alpha, float beta){

    cout << "-------------------------------------------------------------------------"<<endl;
    cout << "Executing single precision GEMM: "<<n<< "x" << m <<endl;
    cout << "-------------------------------------------------------------------------"<<endl;

    FBLASEnvironment fb =FBLASEnvironment (program_path,json_path);
    cl::CommandQueue queue;
    cl::Context context=fb.get_context();
    cl::Device device=fb.get_device();
    IntelFPGAOCLUtils::createCommandQueue(context,device,queue);

    std::string routine_name="sgemm";
    float *A,*B,*C,*fpgaC;

    posix_memalign ((void **)&A, IntelFPGAOCLUtils::AOCL_ALIGNMENT, n*k*sizeof(float));
    posix_memalign ((void **)&B, IntelFPGAOCLUtils::AOCL_ALIGNMENT, k*m*sizeof(float));
    posix_memalign ((void **)&C, IntelFPGAOCLUtils::AOCL_ALIGNMENT, n*m*sizeof(float));
    posix_memalign ((void **)&fpgaC, IntelFPGAOCLUtils::AOCL_ALIGNMENT, n*m*sizeof(float));

    generate_float_matrix(A,n,k);
    generate_float_matrix(B,k,m);
    generate_float_matrix(C,n,m);
    //save C for check
    float *Cblas;
    posix_memalign ((void **)&Cblas, IntelFPGAOCLUtils::AOCL_ALIGNMENT, n*m*sizeof(float));
    memcpy(Cblas,C,n*m*sizeof(float));

    cl::Buffer input_A(context, CL_MEM_READ_ONLY|CL_CHANNEL_1_INTELFPGA, n*k* sizeof(float));
    cl::Buffer input_B(context,CL_MEM_READ_ONLY|CL_CHANNEL_2_INTELFPGA,k*m*sizeof(float));
    cl::Buffer input_output_C(context,CL_MEM_READ_WRITE|CL_CHANNEL_3_INTELFPGA,n*m*sizeof(float));

    queue.enqueueWriteBuffer(input_A,CL_TRUE,0,n*k*sizeof(float),A);
    queue.enqueueWriteBuffer(input_B,CL_TRUE,0,k*m*sizeof(float),B);
    queue.enqueueWriteBuffer(input_output_C,CL_TRUE,0,n*m*sizeof(float),C);

    timestamp_t start_t = current_time_usecs();
    fb.sgemm(routine_name,FBLAS_NO_TRANSPOSED,FBLAS_NO_TRANSPOSED,n,m,k,alpha,input_A,k,input_B,m,beta,input_output_C,m);
    fpga_time = current_time_usecs() - start_t;
    queue.enqueueReadBuffer(input_output_C,CL_TRUE,0,n*m*sizeof(float),fpgaC);


    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,n,m,k,alpha,A,k,B,m,beta,Cblas,m);

    //print_float_matrix(C,n,m);
    const double flteps = 1e-4;
    bool ok=true;

    //    Measure error by considering nrm_inf
    double nrminf_diff=0, nrminf_orig=0;
    double error;

    for(int i=0;i < n; i++)
    {
        double nrminf=0, nrminf_o=0;
        for(int j=0; j<m;j++)
        {
            nrminf+=fabs(fpgaC[i*m+j]-Cblas[i*m+j]);
            nrminf_o+=fabs(Cblas[i*m+j]);
        }
        if(nrminf>nrminf_diff){
            nrminf_diff=nrminf;
        }
        if(nrminf_o>nrminf_orig){
            nrminf_orig=nrminf_o;
        }
    }
    if((nrminf_diff==0 && nrminf_orig ==0) || nrminf_orig==0)
        error=0;
    else
        error=nrminf_diff/nrminf_orig;
    printf("[Measured Error: %f]\n",error);

    ok=(error<flteps);
    if(ok)
        std::cout << "Computation Ok!" <<std::endl;
    else
        std::cout << "ERRORR!!!!!!! " << error<<std::endl;
}


void doublePrecision(std::string program_path, std::string json_path, int n, int k, int m, timestamp_t &fpga_time, double alpha, double beta){

    cout << "-------------------------------------------------------------------------"<<endl;
    cout << "Executing double precision GEMM: "<<n<< "x" << m <<endl;
    cout << "-------------------------------------------------------------------------"<<endl;

    FBLASEnvironment fb =FBLASEnvironment (program_path,json_path);
    cl::CommandQueue queue;
    cl::Context context=fb.get_context();
    cl::Device device=fb.get_device();
    IntelFPGAOCLUtils::createCommandQueue(context,device,queue);

    std::string routine_name="dgemm";
    double *A,*B,*C,*fpgaC;

    posix_memalign ((void **)&A, IntelFPGAOCLUtils::AOCL_ALIGNMENT, n*k*sizeof(double));
    posix_memalign ((void **)&B, IntelFPGAOCLUtils::AOCL_ALIGNMENT, k*m*sizeof(double));
    posix_memalign ((void **)&C, IntelFPGAOCLUtils::AOCL_ALIGNMENT, n*m*sizeof(double));
    posix_memalign ((void **)&fpgaC, IntelFPGAOCLUtils::AOCL_ALIGNMENT, n*m*sizeof(double));

    generate_double_matrix(A,n,k);
    generate_double_matrix(B,k,m);
    generate_double_matrix(C,n,m);
    //save C for check
    double *Cblas;
    posix_memalign ((void **)&Cblas, IntelFPGAOCLUtils::AOCL_ALIGNMENT, n*m*sizeof(double));
    memcpy(Cblas,C,n*m*sizeof(double));

    cl::Buffer input_A(context, CL_MEM_READ_ONLY|CL_CHANNEL_1_INTELFPGA, n*k* sizeof(double));
    cl::Buffer input_B(context,CL_MEM_READ_ONLY|CL_CHANNEL_2_INTELFPGA,k*m*sizeof(double));
    cl::Buffer input_output_C(context,CL_MEM_READ_WRITE|CL_CHANNEL_3_INTELFPGA,n*m*sizeof(double));

    queue.enqueueWriteBuffer(input_A,CL_TRUE,0,n*k*sizeof(double),A);
    queue.enqueueWriteBuffer(input_B,CL_TRUE,0,k*m*sizeof(double),B);
    queue.enqueueWriteBuffer(input_output_C,CL_TRUE,0,n*m*sizeof(double),C);

    timestamp_t start_t = current_time_usecs();
    fb.dgemm(routine_name,FBLAS_NO_TRANSPOSED,FBLAS_NO_TRANSPOSED,n,m,k,alpha,input_A,k,input_B,m,beta,input_output_C,m);
    fpga_time = current_time_usecs() - start_t;
    queue.enqueueReadBuffer(input_output_C,CL_TRUE,0,n*m*sizeof(double),fpgaC);


    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,n,m,k,alpha,A,k,B,m,beta,Cblas,m);

    //print_float_matrix(C,n,m);
    const double flteps = 1e-6;
    bool ok=true;

    //    Measure error by considering nrm_inf
    double nrminf_diff=0, nrminf_orig=0;
    double error;

    for(int i=0;i < n; i++)
    {
        double nrminf=0, nrminf_o=0;
        for(int j=0; j<m;j++)
        {
            nrminf+=fabs(fpgaC[i*m+j]-Cblas[i*m+j]);
            nrminf_o+=fabs(Cblas[i*m+j]);
        }
        if(nrminf>nrminf_diff){
            nrminf_diff=nrminf;
        }
        if(nrminf_o>nrminf_orig){
            nrminf_orig=nrminf_o;
        }
    }
    if((nrminf_diff==0 && nrminf_orig ==0) || nrminf_orig==0)
        error=0;
    else
        error=nrminf_diff/nrminf_orig;
    printf("[Measured Error: %f]\n",error);

    ok=(error<flteps);
    if(ok)
        std::cout << "Computation Ok!" <<std::endl;
    else
        std::cout << "ERRORR!!!!!!! " << error<<std::endl;
}


int main(int argc, char *argv[])
{
    //command line argument parsing
    if(argc<15)
    {
        cerr << "Matrix multiplication A*B where A is NxK and B is KxM " <<endl;
        cerr << "Usage: "<< argv[0]<<" -b <binary file>  -j <generated_json> -n <length> -k <length> -m <length> -a <alpha> -c <beta> -p <precision: float/double> "<<endl;
        exit(-1);
    }
    int c;
    int n,m,p;
    bool double_precision=false;
    float alpha=1,beta=1;
    std::string program_path, json_path;
    while ((c = getopt (argc, argv, "n:m:k:p:b:a:c:j:")) != -1)
        switch (c)
        {
            case 'n':
                n=atoi(optarg);
                break;
            case 'm':
                m=atoi(optarg);
                break;
            case 'a':
                alpha=atof(optarg);
                break;
            case 'c':
                beta=atof(optarg);
                break;
            case 'k':
                p=atoi(optarg);
                break;
            case 'b':
                program_path=std::string(optarg);
                break;
            case 'j':
                json_path=std::string(optarg);
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
                cerr << "Usage: "<< argv[0]<<" -n <length of the vectors> -p <single/double>"<<endl;
                exit(-1);
        }
    std::vector<double> times;
    long data_bytes;
    timestamp_t fpga_time;
    if(double_precision)
        doublePrecision(program_path,json_path,n,p,m,fpga_time, alpha, beta);
    else
        singlePrecision(program_path,json_path,n,p,m,fpga_time, alpha, beta);



    double gops=((double)(m)*n*(2*p-1))/1000000000;
    std::cout << "FPGA Computation time: " << fpga_time << " usecs"<<std::endl;
    std::cout << "FPGA GOps/s: " << gops/((fpga_time)/1000000.0)<<std::endl;


    return 0;
}
