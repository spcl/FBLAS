/**
   Test program for the streaming_gemv module (version 1, i.e. A non trasposed, rowstramed and tiles by rows)

    Please note: for this test we want to generate data on device. Therefore the input sizes must be
    a multiple of the tile size

   The kernels are executed multiple times:
    - for each execution it takes the time using the OpenCL Profiling info command, considering
        the start and the end of a kernel. The execution time is takes that elapses between the start
        of the first kernel to the end of the last one.
    - it outputs in a file all the summary metrics as well as all the measured timings

 */


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string>
#include <iostream>
#include <fstream>
#include <cblas.h>
#include "CL/opencl.h"
#include "../../include/utils/ocl_utils.hpp"
#include "../../include/utils/utils.hpp"
#include "../../include/utils/test.hpp"
#define CHECK
using namespace std;

template <typename T>
void evaluate(std::string program_path, size_t n,  size_t m, T alpha, T beta, int tile_n, int tile_m, std::vector<double> &times, int runs)
{
    std::cout << " alpha= " <<alpha<< ", beta= "<<beta<<std::endl;

    cl::Platform platform;
    cl::Device device;
    cl::Context context;
    cl::Program program;
    timestamp_t comp_start;
    IntelFPGAOCLUtils::initOpenCL(platform,device,context,program,program_path);
    std::vector<std::string> kernel_names;
    if (std::is_same<T, float>::value)
        kernel_names={"sgemv_generate_x","sgemv_generate_y","sgemv_generate_matrix","sgemv","sgemv_write_vector"};
    else
        kernel_names={"dgemv_generate_x","dgemv_generate_y","dgemv_generate_matrix","dgemv","dgemv_write_vector"};
    std::vector<cl::Kernel> kernels;
    std::vector<cl::CommandQueue> queues;
    const int num_kernels=kernel_names.size();
    IntelFPGAOCLUtils::createCommandQueues(context,device,queues,num_kernels);
    IntelFPGAOCLUtils::createKernels(program,kernel_names,kernels);

    int len_y= n;
    int len_x= m;
    T *fpga_res;

    posix_memalign ((void **)&fpga_res, IntelFPGAOCLUtils::AOCL_ALIGNMENT, len_y*sizeof(T));
    cl::Buffer output(context, CL_MEM_READ_WRITE,len_y* sizeof(T));
    int x_repetitions=ceil((float)(len_y)/tile_n);
    int y_repetitions=1;
    int lda=m;

    int one=1;
    //generate x
    kernels[0].setArg(0, sizeof(unsigned int),&len_x);
    kernels[0].setArg(1, sizeof(unsigned int),&x_repetitions);

    //generate y
    kernels[1].setArg(0, sizeof(unsigned int),&len_y);
    kernels[1].setArg(1, sizeof(unsigned int),&y_repetitions);

    //generate matrix
    kernels[2].setArg(0, sizeof(int),&n);
    kernels[2].setArg(1, sizeof(int),&m);

    //sgemv
    kernels[3].setArg(0, sizeof(int),&one);
    kernels[3].setArg(1, sizeof(int),&n);
    kernels[3].setArg(2, sizeof(int),&m);
    kernels[3].setArg(3, sizeof(T),&alpha);
    kernels[3].setArg(4, sizeof(T),&beta);


    //sink
    kernels[4].setArg(0,sizeof(cl_mem), &output);
    kernels[4].setArg(1,sizeof(int),&len_y);
    kernels[4].setArg(2,sizeof(int),&tile_n);

    //enable profiling (needed in 19.1)

    for(int r=0;r<runs;r++)
    {
        cl::Event events[5];
        for(int i=0;i<num_kernels;i++)
        {
            queues[i].enqueueTask(kernels[i],nullptr,&events[i]);
            queues[i].flush();
        }

        //wait
        for(int i=0;i<num_kernels;i++)
        	queues[i].finish();
        ulong min_start, max_end=0;
        ulong end;
        ulong start;
        for(int i=0;i<num_kernels;i++)
        {
            events[i].getProfilingInfo<ulong>(CL_PROFILING_COMMAND_START,&start);
            events[i].getProfilingInfo<ulong>(CL_PROFILING_COMMAND_END,&end);
            if(i==0)
                min_start=start;
            if(start<min_start)
                min_start=start;
            if(end>max_end)
                max_end=end;
        }
        times.push_back((double)((max_end-min_start)/1000.0f));

    }
    //since the kernels will not save the result, is useless to check correctness
    //copy back
    queues[0].enqueueReadBuffer(output,CL_TRUE,0,len_y*sizeof(T),fpga_res);

#if defined(CHECK)
    //NOTE: this assumes a particular way of generating data
    //Also, consider that for very large input size, precision errors could occur
    T *A,*x,*y;
    A=new T[n*m]();
    x=new T[len_x]();
    y=new T[len_y]();

    for(int i=0;i<n;i++)
        for(int j=0;j<m;j++)
            A[i*m+j]=i;
    for(int j=0;j<len_x;j++)
        x[j]=j;
    for(int j=0;j<len_y;j++)
        y[j]=j;

    double flteps;

    //NOTE THAT THIS MATRIX IS STORED BY ROW
    //CBLAS_ORDER ord= row_streamed? CblasRowMajor : CblasColMajor;
    comp_start=current_time_usecs();
    if (std::is_same<T, float>::value){
        cblas_sgemv(CblasRowMajor,CblasNoTrans,n,m,alpha,(float *)A,m,(float *)x,1,beta,(float *)y,1);
        flteps= 1e-4;
    }
    else{
        cblas_dgemv(CblasRowMajor,CblasNoTrans,n,m,alpha,(double *)A,m,(double *)x,1,beta,(double *)y,1);
        flteps= 1e-6;
    }
    bool ok=true;
    //for(int i=0;i<len_y;i++)//orig
    for(int i=0;i<len_y;i++)
    {
        if(!test_equals(fpga_res[i],y[i],flteps)) //orig
        {
            std::cout <<"["<<i<<"] "<< fpga_res[i]<<"\t"<<y[i]<<std::endl;
            ok=false;
        }
    }
    if(ok)
        cout << "Result verified!"<<endl;
    else
        cout << "Result not correct!!! "<<endl;
#endif

}


int main(int argc, char *argv[])
{
    //command line argument parsing
    if(argc<17)
    {
    	cerr << "Matrix-vector multiplication Ax where A is NxM and B is a vector of M elements " <<endl;
        cerr << "Usage: "<< argv[0]<<"-b <binary file> -n <row of A> -m <column of A> -a <alpha> -c <beta>  -k <length of tile TN> -j <length of tile TM>  -r <runs> [-p <precision>]"<<endl;
    	exit(-1);
    }

    int c;
    int n,m;
    int tile_n,tile_m,runs;
    bool double_precision;
    double alpha, beta;
    std::string program_path;
    while ((c = getopt (argc, argv, "n:m:p:b:c:a:k:j:r:")) != -1)
	switch (c)
	{
	    case 'b':
		program_path=std::string(optarg);
		break;
	    case 'n':
		n=atoi(optarg);
		break;
	    case 'm':
		m=atoi(optarg);
		break;
	    case 'p':
		{
		    std::string str=optarg;
		    if(str=="single")
			double_precision=false;
		    else
			if(str=="double")
			    double_precision=true;
			else
			{
			    cerr << "Unrecognized option: " <<optarg<<endl;
			    exit(-1);
			}
		}
		break;
	    case 'k':
    		tile_n=atoi(optarg);
    		break;
	    case 'j':
    		tile_m=atoi(optarg);
    		break;
	    case 'a':
		  alpha = atof(optarg);
		  break;
	    case 'c':
		  beta=atof(optarg);
		  break;
	    case 'r':
            runs=atoi(optarg);
            break;

	    default:
		cerr << "Usage: "<< argv[0]<<" -n <length of the vectors> -p <single/double>"<<endl;
		exit(-1);
	}
    cout << "Matrix: " << n << " x "<<m<<". Tiles: "<<tile_n<<" x "<<tile_m<<endl;
    timestamp_t  cpu_time,fpga_time;
    long data_bytes;
    std::vector<double> times;
    if(!double_precision)
        evaluate<float>(program_path,n,m,alpha,beta,tile_n,tile_m,times,runs);
    else
        evaluate<double>(program_path,n,m,alpha,beta,tile_n,tile_m,times,runs);


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

    double computation_gops=(((double)(2.0f*n*m+2*n))/(1000000000.0)); //for each element we perform a multiplication and an add. We have also to multiply the element of y
    double measured_gops=computation_gops/((mean)/1000000.0);
    cout << "FPGA Computation time (usec): " << mean << " (sttdev: " << stddev<<")"<<endl;
    std::cout << "FPGA GOps/s: " << measured_gops<<std::endl;



    //save the info into output file
    ofstream fout("output.dat");
    fout << "#N = "<<n<<", Runs = "<<runs<<endl;
    fout << "#Average Computation time (usecs): "<<mean<<endl;
    fout << "#Standard deviation (usecs): "<<stddev<<endl;
    fout << "#GOPS/s: "<<measured_gops<<endl;
    fout << "#Execution times (usecs):"<<endl;
    for(auto t:times)
        fout << t << endl;
    fout.close();

    return 0;
}
