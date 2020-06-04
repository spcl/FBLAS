// Axpydot


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
    if(argc<7)
    {
        cerr << "Usage: "<< argv[0]<<"-n <length of the vectors> -a <alpha> -r <num runs> -p <precision float/double>"<<endl;
        exit(-1);
    }

    int c;
    int n, runs;
    double alpha;
    bool double_precision;
    while ((c = getopt (argc, argv, "n:a:b:r:p:")) != -1)
        switch (c)
        {
            case 'n':
                n=atoi(optarg);
                break;
            case 'r':
                runs=atoi(optarg);
                break;
            case 'a':
                alpha=atof(optarg);
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
    std::vector<double> times; //used for the execution with a given par degree
    std::vector<double> comp_times, stddevs; //used to keep track for the different par degree
    std::vector<double> min_times; //the execution times with the best par degree


    mkl_set_dynamic(0);

    //allocate data
    void *u,*w,*v,*z;
    size_t data_size = (double_precision) ? sizeof(double) : sizeof(float);
    posix_memalign ((void **)&u, 64, n*sizeof(data_size));
    posix_memalign ((void **)&w, 64, n*sizeof(data_size));
    posix_memalign ((void **)&v, 64, n*sizeof(data_size));
    posix_memalign ((void **)&z, 64, n*sizeof(data_size));

    if(double_precision){
        cout << "Double precision version." << endl;
        generate_vector<double>((double *)u,n);
        generate_vector<double>((double *)w,n);
        generate_vector<double>((double *)v,n);    
    }
    else{
        cout << "Single precision version." << endl;
        generate_vector<float>((float *)u,n);
        generate_vector<float>((float *)w,n);
        generate_vector<float>((float *)v,n);    
    }

    

    double min_mean=10000000;
    int best_par_degree=0;
    #if defined(POWER_MEASUREMENT)
    //for measuring it we need to consider only one par degree (don't know how to reset these)
    Mammut m;
    Energy* energy = m.getInstanceEnergy();
    CounterCpus* counterCpus = (CounterCpus*) energy->getCounter();
    if(!counterCpus){
      cout << "Cpu counters not present on this machine." << endl;
      return -1;
    }
    double ram_fin, cpu_fin;
    std::vector<double> cpu_watt, ram_watt;
    #endif
    volatile float beta;
    volatile double d_beta;
    for(int p=MIN_THREADS;p<=MAX_THREADS;p++)
    {
        mkl_set_num_threads_local(p);
        times.clear();
        #if defined(POWER_MEASUREMENT)
        counterCpus->reset();
        #endif
        //run experiments
        timestamp_t comp_start,cpu_time;
        timestamp_t power_time_start=current_time_usecs();

        for(int i=0;i<runs;i++)
        {
            comp_start=current_time_usecs();
            asm volatile("": : :"memory");
            if(double_precision){
                cblas_dcopy(n,(double *)w,1,(double *)z,1);
                cblas_daxpy(n,-alpha,(double *)v,1,(double *)z,1);    
                d_beta = cblas_ddot(n,(double *)z,1,(double *)u,1);

            }
            else{
                cblas_scopy(n,(float *)w,1,(float *)z,1);
                cblas_saxpy(n,-alpha,(float *)v,1,(float *)z,1);    
                beta = cblas_sdot(n,(float *)z,1,(float *)u,1);
            }
            
            
            
            asm volatile("": : :"memory");
            times.push_back(current_time_usecs()-comp_start);
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
        //use corrected stddev
        stddev=sqrt(stddev/(runs-1));

        comp_times.push_back(mean);
        stddevs.push_back(stddev);
        if(mean<min_mean)
        {
            min_mean=mean;
            min_times=times;
            best_par_degree=p;
        }
    }

    for(int p=1;p<=MAX_THREADS;p++)
        cout << "["<<p<< "] : time (usec): " <<comp_times[p-1]<< " (stddev: " << stddevs[p-1]<<")"<<endl;
    cout << "Best par degree: "<<best_par_degree<<endl;
    double conf_interval_95=1.96*stddevs[best_par_degree-1]/sqrt(runs);
    double conf_interval_99=2.58*stddevs[best_par_degree-1]/sqrt(runs);
    cout << "Average Computation time (usecs): "<<comp_times[best_par_degree-1]<<endl;
    cout << "Standard deviation (usecs): "<<stddevs[best_par_degree-1]<<endl;
    cout << "Conf interval 95: "<<conf_interval_95<<endl;
    cout << "Conf interval 99: "<<conf_interval_99<<endl;
    cout << "Conf interval 99 within " <<conf_interval_99/comp_times[best_par_degree-1]*100<<"% from mean" <<endl;
    #if defined(POWER_MEASUREMENT)
    cout << "\t Power: "<<cpu_watt[best_par_degree-1]+ram_watt[best_par_degree-1] << " (CPU: "<< cpu_watt[best_par_degree-1]<< ", RAM: "<< ram_watt[best_par_degree-1]<<")"<<endl;
    #endif




    return 0;
}
