// BICG: attention it split the work in two, so the max thread number should be even


// Execution range: change accordingly
#define MAX_THREADS 10
#define MIN_THREADS 1

#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <mkl.h>
#include <math.h>
#include <omp.h>
#include "../../include/utils/utils.hpp"
#include "../../include/utils/data_generators.hpp"

#if defined(POWER_MEASUREMENT)
#include <mammut/mammut.hpp>
using namespace mammut;
using namespace mammut::energy;
#endif
using namespace std;


int main(int argc, char *argv[])
{

    //command line argument parsing
    if(argc<9)
    {
        cerr << "Usage: "<< argv[0]<<" -n <n> -n <m> -r <num runs> -p <precision float/double>"<<endl;
        exit(-1);
    }

    int c;
    int n, m, runs;
    bool double_precision;
    while ((c = getopt (argc, argv, "n:m:r:p:")) != -1)
        switch (c)
        {
            case 'n':
                n=atoi(optarg);
                break;
            case 'r':
                runs=atoi(optarg);
                break;
            case 'm':
                m=atoi(optarg);
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
                cerr << "Usage: "<< argv[0]<<" -n <n> -n <m> -r <num runs> -p <precision float/double>"<<endl;
                exit(-1);
        }
    std::vector<double> times; //used for the execution with a given par degree
    std::vector<double> comp_times, stddevs; //used to keep track for the different par degree
    std::vector<double> min_times; //the execution times with the best par degree


    mkl_set_dynamic(0);

    //allocate data
    void *A,*p,*r,*q,*s;
    size_t data_size = (double_precision) ? sizeof(double) : sizeof(float);
    posix_memalign ((void **)&A, 64, n*m*data_size);
    posix_memalign ((void **)&p, 64, m*data_size);
    posix_memalign ((void **)&q, 64, n*data_size);
    posix_memalign ((void **)&s, 64, m*data_size);
    posix_memalign ((void **)&r, 64, n*data_size);

    if(double_precision){
        cout << "Double precision version." << endl;
        generate_vector<double>((double *)p,m);
        generate_vector<double>((double *)r,n);
        generate_matrix<double>((double *)A,n,m);

    }
    else{
        cout << "Single precision version." << endl;
        generate_vector<float>((float *)p,m);
        generate_vector<float>((float *)r,n);
        generate_matrix<float>((float *)A,n,m);
    }

    double min_mean=10000000;
    int best_par_degree=0;
    #if defined(POWER_MEASUREMENT)
    //for measuring it we need to consider only one par degree (don't know how to reset these)
    Mammut mam;
    Energy* energy = mam.getInstanceEnergy();
    CounterCpus* counterCpus = (CounterCpus*) energy->getCounter();
    if(!counterCpus){
      cout << "Cpu counters not present on this machine." << endl;
      return -1;
    }
    double ram_fin, cpu_fin;
    std::vector<double> cpu_watt, ram_watt;
    #endif

    for(int pd=1;pd<=MAX_THREADS/2;pd++)
    {
        times.clear();
        #if defined(POWER_MEASUREMENT)
        counterCpus->reset();
        #endif
        timestamp_t comp_start,cpu_time;
        timestamp_t power_time_start=current_time_usecs();

        //run experiments
        omp_set_nested(1);
        for(int i=0;i<runs;i++)
        {
            comp_start=current_time_usecs();
            asm volatile("": : :"memory");
            #pragma omp parallel num_threads(2)
            {
                int tid=omp_get_thread_num();
                if (tid ==0)
                {
                    mkl_set_num_threads(pd);
                    if(double_precision)
                        cblas_dgemv(CblasRowMajor,CblasNoTrans,n,m,1,(double *)A,m,(double *)p,1,0,(double *)q,1);
                    else
                        cblas_sgemv(CblasRowMajor,CblasNoTrans,n,m,1,(float *)A,m,(float *)p,1,0,(float *)q,1);
                }
                else
                {
                    mkl_set_num_threads(pd);
                    if(double_precision)
                        cblas_dgemv(CblasRowMajor,CblasTrans,n,m,1,(double *)A,m,(double *)r,1,0,(double *)s,1);
                    else
                        cblas_sgemv(CblasRowMajor,CblasTrans,n,m,1,(float *)A,m,(float *)r,1,0,(float *)s,1);
                }

                asm volatile("": : :"memory");
                times.push_back(current_time_usecs()-comp_start);
            }
        }
         #if defined(POWER_MEASUREMENT)
        //we are measuring the watts
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
        stddev=sqrt(stddev/(runs-1));

        comp_times.push_back(mean);
        stddevs.push_back(stddev);
        if(mean<min_mean)
        {
            min_mean=mean;
            min_times=times;
            best_par_degree=pd*2;
        }

    }

    for(int p=1;p<=MAX_THREADS/2;p++)
        cout << "["<<p*2<< "] : time (usec): " <<comp_times[p-1]<< " (stddev: " << stddevs[p-1]<<")"<<endl;
    cout << "Best par degree: "<<best_par_degree<<endl;
    double conf_interval_95=1.96*stddevs[best_par_degree/2-1]/sqrt(runs);
    double conf_interval_99=2.58*stddevs[best_par_degree/2-1]/sqrt(runs);
    cout << "Average Computation time (usecs): "<<comp_times[best_par_degree/2-1]<<endl;
    cout << "Standard deviation (usecs): "<<stddevs[best_par_degree/2-1]<<endl;
    cout << "Conf interval 95: "<<conf_interval_95<<endl;
    cout << "Conf interval 99: "<<conf_interval_99<<endl;
    cout << "Conf interval 99 within " <<conf_interval_99/comp_times[best_par_degree/2-1]*100<<"% from mean" <<endl;
    #if defined(POWER_MEASUREMENT)
    cout << "\t Power: "<<cpu_watt[best_par_degree/2-1]+ram_watt[best_par_degree/2-1] << " (CPU: "<< cpu_watt[best_par_degree/2-1]<< ", RAM: "<< ram_watt[best_par_degree/2-1]<<")"<<endl;
    #endif

    return 0;
}
