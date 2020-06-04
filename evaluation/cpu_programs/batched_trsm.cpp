/**
 * Batched TRSM
 */

// Execution range: change according to your architecture
#define MAX_THREADS 10
#define MIN_THREADS 1
#include <unistd.h>
#include <stdio.h>
#include <string>
#include <iostream>
#include <fstream>
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

     if(argc<13)
    {
        cerr << "Multiple Right Hand side solver AX=B where A is NxN matrix (lower), X and B are matrices of NxM elements.\nVersion with lower matrix, non transposed, left side " <<endl;
        cerr << "Usage: "<< argv[0]<<" -a <alpha> -n <n size> -m <m size> -l <num_trsm> -r <num_runs> -p <precision float/double>"<<endl;
        exit(-1);
    }

    srand(time(NULL));
    int c;
    int N,M;
    float alpha;
    bool double_precision;
    int ntrsm =1;
    int runs =1 ;

    std::string program_path;
    while ((c = getopt (argc, argv, "a:n:m:f:l:t:r:p:")) != -1)
    switch (c)
    {
        case 'a':
            alpha=atof(optarg);
            break;
        case 'n':
            N=atoi(optarg);
            break;
        case 'm':
            M=atoi(optarg);
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
        cerr << "Usage: "<< argv[0]<<" -a <alpha> -n <n size> -n <m size> -l <num_trsm> -r <num_runs> -p <precision float/double>"<<endl;
        exit(-1);
    }


    const unsigned int align=64;
   


    //allocate and generate A and B
    void *A,*B,*res;
    size_t data_size=double_precision ? sizeof(double):sizeof(float);

    int size_A=N;
    int lda=N;
    posix_memalign ((void **)&A, align, ntrsm*size_A*size_A*data_size);
    posix_memalign ((void **)&res, align, ntrsm*N*M*data_size);
    posix_memalign ((void **)&B, align, ntrsm*N*M*data_size);


    for(int i = 0; i<ntrsm; i++){
        if(double_precision){
            generate_matrix<double>((double *)A+i*(size_A*size_A),size_A,size_A);
            generate_matrix<double>((double *)B+i*(N*M),N,M);
        }
        else{
            generate_matrix<float>((float *)A+i*(size_A*size_A),size_A,size_A);
            generate_matrix<float>((float *)B+i*(N*M),N,M);
        }
    }



    std::vector<double> times; //used for the execution with a given par degree
    std::vector<double> comp_times, stddevs; //used to keep track for the different par degree
    std::vector<double> min_times; //the execution times with the best par degree
    double min_mean=10000000;

    mkl_set_dynamic(0);

    std::cout << "Performing " << ntrsm << " trsm" <<std::endl;

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


    float    alpha_mkl[GRP_COUNT] = {alpha};
    double    d_alpha_mkl[GRP_COUNT] = {alpha};

    void   *a_array[ntrsm];
    void   *b_array[ntrsm];
    for(int i=0;i<ntrsm;i++){
        if(double_precision){
            a_array[i]=&((double*)A)[i*size_A*size_A];
            b_array[i]=&((double *)B)[i*N*M];
        }
        else{
            a_array[i]=&((float*)A)[i*size_A*size_A];
            b_array[i]=&((float *)B)[i*N*M];
        }
    }


    MKL_INT    size_per_grp[GRP_COUNT] = {ntrsm};

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
            // Call 
            if (double_precision){
                cblas_dtrsm_batch (
                    CblasRowMajor,
                    side_mkl,
                    uplo_mkl,
                    transA_mkl,
                    diag_mkl,
                    m_mkl,
                    n_mkl,
                    d_alpha_mkl,
                    (const double **)a_array,
                    lda_mkl,
                    (double **)b_array,
                    ldb_mkl,
                    GRP_COUNT,
                    size_per_grp
                );
            }
            else{
                cblas_strsm_batch (
                    CblasRowMajor,
                    side_mkl,
                    uplo_mkl,
                    transA_mkl,
                    diag_mkl,
                    m_mkl,
                    n_mkl,
                    alpha_mkl,
                    (const float **)a_array,
                    lda_mkl,
                    (float **)b_array,
                    ldb_mkl,
                    GRP_COUNT,
                    size_per_grp
                );
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
