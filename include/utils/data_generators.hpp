#ifndef DATA_GENERATORS_HPP
#define DATA_GENERATORS_HPP
#include <iostream>

void generate_float_lower_matrix(float *A, int N)
{
    //    cout<< "Lower"<<endl;
    //generate data
    for(int i=0;i<N;i++)
    {
        for(int j=0;j<N;j++)
        {
            if(j<=i)
                //
                A[i*N+j]=A[i*N+j] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/100.0));
            else
                A[i*N+j]=0;
        }
        //x[i]=static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/100.0));

    }
}

void generate_double_lower_matrix(double *A, int N)
{
    //    cout<< "Lower"<<endl;
    //generate data
    for(int i=0;i<N;i++)
    {
        for(int j=0;j<N;j++)
        {
            if(j<=i)
                //		A[i*N+j] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/100.0));
                A[i*N+j]=i+1;
            else
                A[i*N+j]=0;
        }
        //x[i]=static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/100.0));

    }
}

void generate_float_upper_matrix(float *A, int N)
{
    //generate data
    for(int i=0;i<N;i++)
    {
        for(int j=0;j<N;j++)
        {
            if(j<i)
                A[i*N+j]=0;
            else
                A[i*N+j]=static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/100.0));

            //	    A[i*N+j] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/100.0));
        }
        //x[i]=static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/100.0));

    }
}

template <typename T>
void generate_matrix(T *A,int N,int M)
{
    for(size_t i=0;i<N;i++)
    {
        for(size_t j=0;j<M;j++)
           A[i*M+j] = static_cast <T> (rand()) / (static_cast <T> (RAND_MAX/100.0));
    }
}

void generate_float_matrix(float *A,int N,int M)
{
    for(size_t i=0;i<N;i++)
    {
        for(size_t j=0;j<M;j++)
           A[i*M+j] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/100.0));
    }
}


void generate_double_matrix(double *A,int N,int M)
{
    for(size_t i=0;i<N;i++)
    {
        for(size_t j=0;j<M;j++)
            A[i*M+j]=static_cast <double> (rand()) / (static_cast <double> (RAND_MAX/10.0));
        //matrix[i][j] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/100.0));
    }
}

template <typename T>
void generate_vector(T *x, int N)
{
    for(int i=0;i<N;i++)
        x[i] = static_cast <T> (rand()) / (static_cast <T> (RAND_MAX/1.0));
}

void generate_float_vector (float *x, int N)
{
    for(int i=0;i<N;i++)
        x[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/1.0));
}


void generate_double_vector (double *x, int N)
{
    for(int i=0;i<N;i++)
        x[i]= static_cast <double> (rand()) / (static_cast <double> (RAND_MAX/1.0));
}

void print_float_matrix(float *A, int N, int M)
{
    std::cout << "----------------------------" <<std::endl;
    for(int i=0;i<N;i++)
    {
        for(int j=0;j<M;j++)
            std::cout << A[i*M+j] << "\t";
        std::cout<<std::endl;
    }
    std::cout << "----------------------------" <<std::endl;
}

void print_float_vector(float *x, int N)
{
    std::cout << "----------------------------" <<std::endl;
    for(int i=0;i<N;i++)
    {
       std::cout << x[i] << "\n";
    }
    std::cout << "----------------------------" <<std::endl;
}
#endif // DATA_GENERATORS_HPP
