// DOT Product


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
#include <cmath>
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
        cerr << "Usage: "<< argv[0]<<" -n <length of the vectors> -p <single/double> -r <num runs>"<<endl;
        exit(-1);
    }

    int c;
    int n, runs;
    bool double_precision;
    std::string program_name;
    while ((c = getopt (argc, argv, "n:p:b:r:")) != -1)
        switch (c)
        {
            case 'n':
                n=atoi(optarg);
                break;
            case 'r':
                runs=atoi(optarg);
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
                break;
            }
            default:
                cerr << "Usage: "<< argv[0]<<" -n <length of the vectors> -p <single/double>"<<endl;
                exit(-1);
        }
    std::vector<double> times; //used for the execution with a given par degree
    std::vector<double> comp_times, stddevs; //used to keep track for the different par degree
    std::vector<double> min_times; //the execution times with the best par degree
   

    float *x,*y;
    double *dx,*dy;

    if(double_precision){
        std::cout << "Executing in double precision. " <<std::endl;
        posix_memalign((void **)&dx, 64, n*sizeof(double));
        posix_memalign((void **)&dy, 64, n*sizeof(double));
        generate_vector<double>(dx,n);
        generate_vector<double>(dy,n);
    }
    else{
        std::cout << "Executing in single precision." <<std::endl;
        posix_memalign((void **)&x, 64, n*sizeof(float));
        posix_memalign((void **)&y, 64, n*sizeof(float));
        generate_vector<float>(y,n);
        generate_vector<float>(x,n);

    }    

    double min_mean=1e9;
    int best_par_degree=0;
    int best_index;
#if defined(POWER_MEASUREMENT)
    Mammut m;
    Energy* energy = m.getInstanceEnergy();
    CounterCpus* counterCpus = (CounterCpus*) energy->getCounter();
    if(!counterCpus){
      cout << "Cpu counters not present on this machine." << endl;
      return -1;
    }
    double cpu= counterCpus->getJoulesCpuAll();
    double core= counterCpus->getJoulesCoresAll();
    double ram= counterCpus->getJoulesDramAll() ;
    double ram_fin, cpu_fin;
    std::vector<double> cpu_watt, ram_watt;

#endif

    for(int p=MIN_THREADS;p<=MAX_THREADS;p++)
    {
        mkl_set_dynamic(0);
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
            volatile float res;
            volatile double d_res;
            timestamp_t start=current_time_usecs();
            asm volatile("": : :"memory");
            if(double_precision)
                d_res = cblas_ddot(n, dx, 1, dy, 1);
            else
                res = cblas_sdot(n, x, 1, y, 1);
            timestamp_t end_t=current_time_usecs()-start;
            asm volatile("": : :"memory");
            if(i>0) //remove the first run
            times.push_back(end_t);
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

    double conf_interval_95=1.96*stddevs[best_index]/sqrt(runs);
    double conf_interval_99=2.58*stddevs[best_index]/sqrt(runs);
    cout << "Best par degree: "<<best_par_degree<<endl;
    cout << "Average Computation time (usecs): "<<comp_times[best_index]<<endl;
    cout << "Standard deviation (usecs): "<<stddevs[best_index]<<endl;
    cout << "Conf interval 99 within " <<conf_interval_99/comp_times[best_index]*100<<"% from mean" <<endl;
#if defined(POWER_MEASUREMENT)    
    cout << "\t Power: "<<cpu_watt[best_index]+ram_watt[best_index] << " (CPU: "<< cpu_watt[best_index]<< ", RAM: "<< ram_watt[best_index]<<")"<<endl;
#endif 
    return 0;
}
