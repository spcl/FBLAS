/**
 * Batched GEMM
 */

// Execution range: change according to your architecture
#define MAX_THREADS 10
#define MIN_THREADS 1

#include <unistd.h>
#include <stdio.h>
#include <string>
#include <iostream>
#include <fstream>
//#include <cblas.h>
#include <mkl.h>
#include <math.h>
#if defined(POWER_MEASUREMENT)
#include <mammut/mammut.hpp>
using namespace mammut;
using namespace mammut::energy;
#endif
#include "../../include/utils/utils.hpp"
#include "../../include/utils/data_generators.hpp"


using namespace std;
int main(int argc, char *argv[])
{

    //command line argument parsing
    if(argc<17)
    {
        cerr << "Batched GEMM " <<endl;
        cerr << "Usage: "<< argv[0]<<" -n <length> -k <length> -m <length> -a <alpha> -c <beta> -l <num_gemms> -r <num runs>  -p <precision float/double>"<<endl;
        exit(-1);
    }
    const unsigned int align=64;
    int c;
    int n,m,k;
    int num_gemm;
    bool double_precision;
    std::string program_path;
    int runs = 1;
    float alpha, beta;
    while ((c = getopt (argc, argv, "n:m:k:a:c:l:r:p:")) != -1)
        switch (c)
        {
            case 'n':
                n=atoi(optarg);
                break;
            case 'm':
                m=atoi(optarg);
                break;
            case 'k':
                k=atoi(optarg);
                break;
            case 'r':
                runs=atoi(optarg);
                break;
            case 'a':
                alpha=atof(optarg);
                break;
            case 'c':
                beta=atof(optarg);
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
                cerr << "Usage: "<< argv[0]<<" -b <binary file> -n <length> -k <length> -m <length>"<<endl;
                exit(-1);
        }

    cout << "Performing multiplication A ("<<n<<" x "<< k<<") * B ("<<k<<" x "<<m<<")"<<endl;

    //init data
    void *A,*B,*C;
    size_t data_size=double_precision ? sizeof(double):sizeof(float);
    posix_memalign ((void **)&A, align, num_gemm*n*k*data_size);
    posix_memalign ((void **)&B, align, num_gemm*k*m*data_size);
    posix_memalign ((void **)&C, align, num_gemm*n*m*data_size);
    for(int i=0;i<num_gemm;i++)
    {   
        if (double_precision)
        {
            generate_matrix<double>((double *)A+i*(n*k),n,k);
            generate_matrix<double>((double *)B+i*(m*k),k,m);
            generate_matrix<double>((double *)C+i*(n*m),n,m);
        }
        else{
            generate_matrix<float>((float *)A+i*(n*k),n,k);
            generate_matrix<float>((float *)B+i*(m*k),k,m);
            generate_matrix<float>((float *)C+i*(n*m),n,m);
        }
    }



    std::vector<double> times; //used for the execution with a given par degree
    std::vector<double> comp_times, stddevs; //used to keep track for the different par degree
    std::vector<double> min_times; //the execution times with the best par degree
    double min_mean=10000000;

    mkl_set_dynamic(0);

    std::cout << "Performing " << num_gemm << " gemms" <<std::endl;

    //#if 0
    //batched gemm
    #define    GRP_COUNT    1

    MKL_INT    m_mkl[GRP_COUNT] = {4};
    MKL_INT    k_mkl[GRP_COUNT] = {4};
    MKL_INT    n_mkl[GRP_COUNT] = {4};

    MKL_INT    lda_mkl[GRP_COUNT] = {4};
    MKL_INT    ldb_mkl[GRP_COUNT] = {4};
    MKL_INT    ldc_mkl[GRP_COUNT] = {4};

    CBLAS_TRANSPOSE    transA[GRP_COUNT] = {CblasNoTrans};
    CBLAS_TRANSPOSE    transB[GRP_COUNT] = {CblasNoTrans};

    float    alpha_mkl[GRP_COUNT] = {alpha};
    float    beta_mkl[GRP_COUNT] = {beta};
    double   d_alpha_mkl[GRP_COUNT] = {alpha};
    double   d_beta_mkl[GRP_COUNT] = {beta};

    void   *a_array[num_gemm];
    void   *b_array[num_gemm];
    void   *c_array[num_gemm];
    for(int i=0;i<num_gemm;i++){
        if(double_precision){
            a_array[i]=&((double *)A)[i*n*m];
            b_array[i]=&((double *)B)[i*n*m];
            c_array[i]=&((double *)C)[i*n*m];
        }
        else{
            a_array[i]=&((float *)A)[i*n*m];
            b_array[i]=&((float *)B)[i*n*m];
            c_array[i]=&((float *)C)[i*n*m];

        }
    }


    MKL_INT    size_per_grp[GRP_COUNT] = {num_gemm};

    int best_par_degree=-1;
    int best_index;
#if defined(POWER_MEASUREMENT)
    //for measuring it we need to consider only one par degree (don't know how to reset these)
    Mammut mammut;
    Energy* energy = mammut.getInstanceEnergy();
    CounterCpus* counterCpus = (CounterCpus*) energy->getCounter();
    if(!counterCpus){
      cout << "Cpu counters not present on this machine." << endl;
      return -1;
    }
    double ram_fin, cpu_fin;
    std::vector<double> cpu_watt, ram_watt;

#endif


     for(int p=MIN_THREADS;p<=MAX_THREADS;p++)
    {
        mkl_set_num_threads_local(p);
        times.clear();
        cout<< "Executing with "<< p << " threads"<<endl;
        #if defined(POWER_MEASUREMENT)
        counterCpus->reset();
        #endif
        //run experiments
        timestamp_t comp_start,cpu_time;
        timestamp_t power_time_start=current_time_usecs();
        for(int i=0;i<=runs;i++)
        {
            timestamp_t start=current_time_usecs();
            asm volatile("": : :"memory");
            // Call cblas_sgemm_batch
            if(double_precision){
                cblas_dgemm_batch (
                CblasRowMajor,
                transA,
                transB,
                n_mkl,
                m_mkl,
                k_mkl,
                d_alpha_mkl,
                (const double **)a_array,
                lda_mkl,
                (const double **)b_array,
                ldb_mkl,
                d_beta_mkl,
                (double **)c_array,
                ldc_mkl,
                GRP_COUNT,
                size_per_grp);
            }
            else{
                cblas_sgemm_batch (
                CblasRowMajor,
                transA,
                transB,
                n_mkl,
                m_mkl,
                k_mkl,
                alpha_mkl,
                (const float **)a_array,
                lda_mkl,
                (const float **)b_array,
                ldb_mkl,
                beta_mkl,
                (float **)c_array,
                ldc_mkl,
                GRP_COUNT,
                size_per_grp);

            }
            timestamp_t end_t=current_time_usecs()-start;
            asm volatile("": : :"memory");
            if(i>0) //remove the first run
            {
                times.push_back(end_t);
            }

        }

#if defined(POWER_MEASUREMENT)
        timestamp_t power_time_end=current_time_usecs()-power_time_start;
        cpu_fin= counterCpus->getJoulesCpuAll();
        ram_fin= counterCpus->getJoulesDramAll();
        cpu_watt.push_back( cpu_fin/(power_time_end/1000000.0));
        ram_watt.push_back( ram_fin/(power_time_end/1000000.0));
#endif
        //compute the average and standard deviation of times
        double mean=0;
        for(auto t:times)
            mean+=t;
        mean/=runs;
        double stddev=0;
        for(auto t:times)
            stddev+=((t-mean)*(t-mean));
        //use corrected stddev
        stddev=sqrt(stddev/(runs-1));

        comp_times.push_back(mean);
        stddevs.push_back(stddev);
        if(mean<min_mean)
        {
            min_mean=mean;
            min_times=times;
            best_par_degree=p;
            best_index=comp_times.size()-1;
        }

    }



    for(int p=MIN_THREADS;p<=MAX_THREADS;p++)
    {
        cout << "["<<p<< "]: time (usec): " <<comp_times[p-1]<< " (stddev: " << stddevs[p-1]<<")"<<endl;
        cout << "\t Power: "<<cpu_watt[p-1]+ram_watt[p-1] << " (CPU: "<< cpu_watt[p-1]<< ", RAM: "<< ram_watt[p-1]<<")"<<endl;
    }
    cout << "Best par degree: "<<best_par_degree<<endl;
    double conf_interval_95=1.96*stddevs[best_index]/sqrt(runs);
    double conf_interval_99=2.58*stddevs[best_index]/sqrt(runs);
    cout << "Average Computation time (usecs): "<<comp_times[best_index]<<endl;
    cout << "Standard deviation (usecs): "<<stddevs[best_index]<<endl;
    cout << "Conf interval 95: "<<conf_interval_95<<endl;
    cout << "Conf interval 99: "<<conf_interval_99<<endl;
    cout << "Conf interval 99 within " <<conf_interval_99/comp_times[best_index]*100<<"% from mean" <<endl;
    #if defined(POWER_MEASUREMENT)
    cout << "\t Power: "<<cpu_watt[best_index]+ram_watt[best_index] << " (CPU: "<< cpu_watt[best_index]<< ", RAM: "<< ram_watt[best_index]<<")"<<endl;
    #endif
    

}
