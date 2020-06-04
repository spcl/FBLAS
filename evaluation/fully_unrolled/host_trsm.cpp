/*
	Fully unrolled TRSM

	Attention: being fully unrolled, the routine works with the matrix sizes given during compilation
*/


#include <stdio.h>
#include <string>
#include <iostream>
#include "CL/opencl.h"
#include <fstream>
// #include <cblas.h>
#include <mkl.h>
#include <math.h>
#include "../../include/utils/ocl_utils.hpp"
#include "../../include/utils/utils.hpp"
#include "../../include/utils/test.hpp"
#include "../../include/utils/data_generators.hpp"
using namespace std;

template <typename T>
void evaluate(std::string program_path, T alpha, int ntrsm, std::vector<double> &times, timestamp_t &cpu_time, int runs){


    int N,M;
    if (std::is_same<T, float>::value){
        cout << "Single precision: working with 4x4 matrices" <<endl;
        N=4;
        M=4;
    }
    else{
        cout << "Double precision: working with 4x4 matrices" <<endl;
        N=4;
        M=4;
    }
    cl_int status;
    cl::Platform  platform;
    cl::Device device;
    cl::Context context;
    cl::Program program;
    std::vector<cl::Kernel> kernels;
    std::vector<cl::CommandQueue> queues;
    std::vector<std::string> kernel_names={"strsm", "read_matrix_A", "read_matrix_B", "write_matrix_B"};
    cout << "Reprogramming device ..."<<endl;
    IntelFPGAOCLUtils::initEnvironment(platform,device,context,program,program_path,kernel_names, kernels,queues);
    cout << "Device reprogrammed."<<endl;

    //allocate and generate A and B
    T *A,*B,*res;
    int size_A=N;
    int lda=N;
    posix_memalign ((void **)&A, IntelFPGAOCLUtils::AOCL_ALIGNMENT, ntrsm*size_A*size_A*sizeof(T));
    posix_memalign ((void **)&res, IntelFPGAOCLUtils::AOCL_ALIGNMENT, ntrsm*N*M*sizeof(T));
    posix_memalign ((void **)&B, IntelFPGAOCLUtils::AOCL_ALIGNMENT, ntrsm*N*M*sizeof(T));


    for(int i = 0; i<ntrsm; i++){
        if (std::is_same<T, float>::value){
            generate_float_matrix((float*)(A+i*(size_A*size_A)),size_A,size_A);
            generate_float_matrix((float*)(B+i*(N*M)),N,M);
        }
        else{
            generate_double_matrix((double *)(A+i*(size_A*size_A)),size_A,size_A);
            generate_double_matrix((double *)(B+i*(N*M)),N,M);
        }
    }

    cl::Buffer input_A(context, CL_MEM_READ_ONLY|CL_CHANNEL_1_INTELFPGA, ntrsm*size_A*size_A * sizeof(T));
    cl::Buffer input_output_B(context, CL_MEM_READ_WRITE|CL_CHANNEL_2_INTELFPGA, ntrsm*N*M* sizeof(T));

    queues[0].enqueueWriteBuffer(input_A,CL_TRUE,0,ntrsm*size_A*size_A*sizeof(T),A);

    kernels[0].setArg(0,sizeof(int),&N);
    kernels[0].setArg(1,sizeof(int),&M);
    kernels[0].setArg(2,sizeof(T),&alpha);
    kernels[0].setArg(3,sizeof(int),&ntrsm);

    //read_A
    kernels[1].setArg(0,sizeof(cl_mem),&input_A);
    kernels[1].setArg(1,sizeof(int),&ntrsm);

    //read_B
    kernels[2].setArg(0,sizeof(cl_mem),&input_output_B);
    kernels[2].setArg(1,sizeof(int),&ntrsm);

    //writeB
    kernels[3].setArg(0,sizeof(cl_mem),&input_output_B);
    kernels[3].setArg(1,sizeof(int),&ntrsm);

    std::cout << "Launching kernel ... " <<std::endl;

    for(int r=0;r<runs;r++)
    {
        queues[0].enqueueWriteBuffer(input_output_B,CL_TRUE,0,ntrsm*N*M*sizeof(T),B);

        timestamp_t fpga_start = current_time_usecs();
        for(int i=0;i<kernels.size();i++)
            queues[i].enqueueTask(kernels[i]);

        //wait
        for(int i=0;i<kernels.size();i++)
            queues[i].finish();
        timestamp_t fpga_time = current_time_usecs() - fpga_start;

        times.push_back((double)(fpga_time));
    }
    //get back the result
    queues[0].enqueueReadBuffer(input_output_B,CL_TRUE,0,ntrsm* N*M*sizeof(T),res);


    //check
    CBLAS_SIDE side=CblasLeft;
    CBLAS_TRANSPOSE transp=CblasNoTrans;
    CBLAS_UPLO uplo=CblasLower;

    std::cout << "MKL executed with 10 threads "<<std::endl;

    mkl_set_dynamic(0);
    mkl_set_num_threads_local(10);


     //batched trsm
    #define    GRP_COUNT    1

    MKL_INT    m_mkl[GRP_COUNT] = {N};
    MKL_INT    n_mkl[GRP_COUNT] = {M};

    MKL_INT    lda_mkl[GRP_COUNT] = {N};
    MKL_INT    ldb_mkl[GRP_COUNT] = {M};

    CBLAS_SIDE side_mkl[GRP_COUNT] = {CblasLeft};
    CBLAS_UPLO uplo_mkl[GRP_COUNT] = {CblasLower};

    CBLAS_TRANSPOSE    transA_mkl[GRP_COUNT] = {CblasNoTrans};
    CBLAS_TRANSPOSE    transB_mkl[GRP_COUNT] = {CblasNoTrans};

    CBLAS_DIAG  diag_mkl[GRP_COUNT] = {CblasNonUnit};


    T    alpha_mkl[GRP_COUNT] = {alpha};

    T   *a_array[ntrsm];
    T   *b_array[ntrsm];

    for(int i=0;i<ntrsm;i++){
        a_array[i]=&A[i*size_A*size_A];
        b_array[i]=&B[i*N*M];
    }


    MKL_INT    size_per_grp[GRP_COUNT] = {ntrsm};

    timestamp_t cpu_start = current_time_usecs();

    if (std::is_same<T, float>::value){
        // Call cblas_strsm_batch
        cblas_strsm_batch (
                CblasRowMajor,
                side_mkl,
                uplo_mkl,
                transA_mkl,
                diag_mkl,
                m_mkl,
                n_mkl,
                (float *)alpha_mkl,
                (const float **) a_array,
                lda_mkl,
                (float **)b_array,
                ldb_mkl,
                GRP_COUNT,
                size_per_grp
        );
    }
    else{
        cblas_dtrsm_batch (
                CblasRowMajor,
                side_mkl,
                uplo_mkl,
                transA_mkl,
                diag_mkl,
                m_mkl,
                n_mkl,
                (double *)alpha_mkl,
                (const double **) a_array,
                lda_mkl,
                (double **)b_array,
                ldb_mkl,
                GRP_COUNT,
                size_per_grp
        );
    }


    cpu_time = current_time_usecs() - cpu_start;

    bool ok=true;
    for(int i=0; i<ntrsm; i++){
        bool r = compare_matrices<T>((res+i*(N*M)),(B+i*(N*M)),N,M,1e-4);
        ok &=r;
    }

    if (ok)
    	cout <<"Ok! "<<endl;
    else
    	cout <<"Error! "<<endl;
}

using namespace std;
int main(int argc, char *argv[])
{
    if(argc<9)
    {
    	cerr << "Multiple Right Hand side solver AX=B where A is NxN matrix (lowert), X and B are matrices of NxM elements.\nVersion with lower matrix, non transposed, left side " <<endl;
        cerr << "Usage: "<< argv[0]<<" -b <binary file> -a <alpha> -r <runs>  -l <num_trsm> -p <precision float/double> "<<endl;
    	exit(-1);
    }

    srand(time(NULL));
    int c;
    int N,M;
    float alpha;
    bool double_precision;
    int ntrsm =1;
    int runs = 10;
    std::string program_path;
    while ((c = getopt (argc, argv, "a:r:b:f:l:t:p:")) != -1)
	switch (c)
	{
        case 'a':
            alpha=atof(optarg);
            break;
	    case 'b':
    		program_path=std::string(optarg);
    		break;
	    case 'r':
    		runs=atoi(optarg);
    		break;
        case 'l':
            ntrsm = atoi(optarg);
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
		cerr << "Usage: "<< argv[0]<<" -b <binary file> -a <alpha> -r <runs>  -l <num_trsm> -p <precision float'double>"<<endl;
		exit(-1);
	}

	std::vector<double> times;
    timestamp_t cpu_time;
    if(double_precision)
        evaluate<double>(program_path,alpha, ntrsm, times, cpu_time, runs );
    else
        evaluate<float>(program_path,alpha, ntrsm, times, cpu_time, runs );


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

     cout << "FPGA Computation time (usec): " << mean << " (sttdev: " << stddev<<")"<<endl;
    cout << "CPU time (usecs): " <<cpu_time<<endl;






}

