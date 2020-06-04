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
    if(argc<11)
    {
        cerr << "Usage: "<< argv[0]<<"-n <length of the vectors> -a <alpha> -c <beta> -r <num runs> -p <precision float/double>"<<endl;
        exit(-1);
    }

    int c;
    int n, runs;
    double alpha,beta;
    bool double_precision;
    std::string program_name;
    while ((c = getopt (argc, argv, "n:a:c:r:p:")) != -1)
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
            case 'c':
                beta=atof(optarg);
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
    void *A,*B,*u1,*u2,*w,*v1,*v2,*z,*y,*x;
    size_t data_size = (double_precision) ? sizeof(double) : sizeof(float);

    posix_memalign ((void **)&A, 64, n*n*data_size);
    posix_memalign ((void **)&B, 64, n*n*data_size);

    posix_memalign ((void **)&u1, 64, n*data_size);
    posix_memalign ((void **)&u2, 64, n*data_size);
    posix_memalign ((void **)&w, 64, n*data_size);
    posix_memalign ((void **)&v1, 64, n*data_size);
    posix_memalign ((void **)&v2, 64, n*data_size);
    posix_memalign ((void **)&z, 64, n*data_size);
    posix_memalign ((void **)&y, 64, n*data_size);
    posix_memalign ((void **)&x, 64, n*data_size);
    if(double_precision)
    {
        generate_vector<double>((double *)u1,n);
        generate_vector<double>((double *)u2,n);
        generate_vector<double>((double *)y,n);
        generate_vector<double>((double *)z,n);
        generate_vector<double>((double *)v1,n);
        generate_vector<double>((double *)v2,n);
        generate_matrix<double>((double *)A,n,n);

    }
    else{
        generate_vector<float>((float *)u1,n);
        generate_vector<float>((float *)u2,n);
        generate_vector<float>((float *)y,n);
        generate_vector<float>((float *)z,n);
        generate_vector<float>((float *)v1,n);
        generate_vector<float>((float *)v2,n);
        generate_matrix<float>((float *)A,n,n);
    }

    double min_mean=10000000;
    int best_par_degree=-1;
    int best_index;
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

    for(int p=MIN_THREADS;p<=MAX_THREADS;p++)
    {
        mkl_set_num_threads(p);
        times.clear();
        cout<< "Executing with "<< p << " threads"<<endl;
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

            if(double_precision)
            {
                cblas_dcopy(n*n,(double *)A,1,(double *)B,1);
                cblas_dger(CblasRowMajor,n,n,1,(double *)u1,1,(double *)v1,1,(double *)B,n);
                cblas_dger(CblasRowMajor,n,n,1,(double *)u2,1,(double *)v2,1,(double *)B,n);
                cblas_dcopy(n,(double *)z,1,(double *)x,1);
                cblas_dgemv(CblasRowMajor,CblasTrans,n,n,beta,(double *)B,n,(double *)y,1,1,(double *)x,1);
                cblas_dgemv(CblasRowMajor,CblasNoTrans,n,n,alpha,(double *)B,n,(double *)x,1,0,(double *)w,1);
            }
            else{
                cblas_scopy(n*n,(float *)A,1,(float *)B,1);
                cblas_sger(CblasRowMajor,n,n,1,(float *)u1,1,(float *)v1,1,(float *)B,n);
                cblas_sger(CblasRowMajor,n,n,1,(float *)u2,1,(float *)v2,1,(float *)B,n);
                cblas_scopy(n,(float *)z,1,(float *)x,1);
                cblas_sgemv(CblasRowMajor,CblasTrans,n,n,beta,(float *)B,n,(float *)y,1,1,(float *)x,1);
                cblas_sgemv(CblasRowMajor,CblasNoTrans,n,n,alpha,(float *)B,n,(float *)x,1,0,(float *)w,1);

            }
            
            asm volatile("": : :"memory");
            times.push_back(current_time_usecs()-comp_start);
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
    double conf_interval_95=1.96*stddevs[best_index]/sqrt(runs);
    double conf_interval_99=2.58*stddevs[best_index]/sqrt(runs);
    cout << "Average Computation time (usecs): "<<comp_times[best_index]<<endl;
    cout << "Standard deviation (usecs): "<<stddevs[best_index]<<endl;
    cout << "Conf interval 95: "<<conf_interval_95<<endl;
    cout << "Conf interval 99: "<<conf_interval_99<<endl;
    cout << "Conf interval 99 within " <<conf_interval_99/comp_times[best_index]*100<<"% from mean" <<endl;
    cout << "\t Power: "<<cpu_watt[best_index]+ram_watt[best_index] << " (CPU: "<< cpu_watt[best_index]<< ", RAM: "<< ram_watt[best_index]<<")"<<endl;


    return 0;
}
