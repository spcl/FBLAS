/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2019 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    Host Api Implementation - Level 2 Routines
*/


#ifndef FBLAS_ENVIRONMENT_LEVEL2_HPP
#define FBLAS_ENVIRONMENT_LEVEL2_HPP

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <math.h>
#include <rapidjson/document.h>
#include "../fblas_environment.hpp"
#include "../utils/ocl_utils.hpp"


/*----------------
 * GEMV
 *---------------*/
template <typename T>
void FBLASEnvironment::gemv(std::string routine_name, FblasTranspose transposed, unsigned int N, unsigned int M, T alpha, cl::Buffer A,
                            unsigned int lda, cl::Buffer x, int incx, T beta, cl::Buffer y, int incy, std::vector<cl::Event> *events_wait_list, cl::Event *event)
{
    auto it= routines_.find(routine_name);
    if(it == routines_.end())
    {
        std::cerr << "There not exist a routine with this given name: "<<routine_name  << std::endl;
        return;
    }
    Routine r=it->second;

    FblasOrder ord=r.order;

    CHECK_TRANS(transposed,r);
    CHECK_INCX(incx,r);
    CHECK_INCY(incy,r);

    unsigned int x_repetitions=1;
    unsigned int y_repetitions=(beta==0)?0:1;
    int order=(ord==FBLAS_ROW_MAJOR)?1:0;
    unsigned int x_size=(transposed==FBLAS_TRANSPOSED)? N:M;
    unsigned int y_size=(transposed==FBLAS_TRANSPOSED)? M:N;
    unsigned int tile_y_size=(transposed==FBLAS_TRANSPOSED)? r.tile_m_size:r.tile_n_size;
    unsigned int tile_x_size=(transposed==FBLAS_TRANSPOSED)? r.tile_n_size:r.tile_m_size;
    //unsigned int lda=(r.lda==0)?M:r.lda;

    if(ord==FBLAS_ROW_MAJOR && transposed==FBLAS_NO_TRANSPOSED) //in this case x must be repeated N/tile_N times
        x_repetitions=ceil((float)(N)/r.tile_n_size);

    if(ord==FBLAS_ROW_MAJOR && transposed==FBLAS_TRANSPOSED)    //in this case x must be repeated M/TILE_M size
        x_repetitions=ceil((float)(M)/r.tile_m_size);

    //Set kernel arguments, according to the routine characteristics
    //gemv

    r.kernels[0].setArg(0, sizeof(int),&order);
    r.kernels[0].setArg(1, sizeof(int),&N);
    r.kernels[0].setArg(2, sizeof(int),&M);
    r.kernels[0].setArg(3, sizeof(T),&alpha);
    r.kernels[0].setArg(4, sizeof(T),&beta);
    //matrix reader
    r.kernels[1].setArg(0, sizeof(cl_mem),&A);
    r.kernels[1].setArg(1, sizeof(int),&N);
    r.kernels[1].setArg(2, sizeof(int),&M);
    r.kernels[1].setArg(3, sizeof(int),&lda);

    //read x and y
    r.kernels[2].setArg(0,sizeof(cl_mem), &x);
    r.kernels[2].setArg(1, sizeof(unsigned int),&x_size);
    r.kernels[2].setArg(2, sizeof(unsigned int),&tile_x_size);
    r.kernels[2].setArg(3, sizeof(unsigned int),&x_repetitions);

    r.kernels[3].setArg(0,sizeof(cl_mem), &y);
    r.kernels[3].setArg(1, sizeof(unsigned int),&y_size);
    r.kernels[3].setArg(2, sizeof(unsigned int),&tile_y_size);
    r.kernels[3].setArg(3, sizeof(unsigned int),&y_repetitions);


    r.kernels[4].setArg(0,sizeof(cl_mem), &y);
    r.kernels[4].setArg(1,sizeof(int),&y_size);  //attention to this
    r.kernels[4].setArg(2,sizeof(int),&tile_y_size);

    //launch kernels: if the routine is non-blocking (i.e. the event parameter is not null)
    //we create the corresponding event when we launch the sink kernel (the last one)

    for(int i=0;i<r.kernels.size()-1;i++)
        r.queues[i].enqueueTask(r.kernels[i],events_wait_list);

    //launch the last one
    r.queues[r.kernels.size()-1].enqueueTask(r.kernels[r.kernels.size()-1],events_wait_list,event);

    if(!event)
        for(int i=0;i<r.kernels.size();i++)
            r.queues[i].finish();
}

void FBLASEnvironment::sgemv(std::string routine_name, FblasTranspose transposed, unsigned int N, unsigned int M, float alpha, cl::Buffer A,
                        unsigned int lda, cl::Buffer x, int incx, float beta, cl::Buffer y, int incy, std::vector<cl::Event> *events_wait_list, cl::Event *event)
{

    //call the generic gemv
    FBLASEnvironment::gemv<float>(routine_name,transposed,N,M,alpha,A,lda,x,incx, beta,y, incy, events_wait_list,event);
}

void FBLASEnvironment::dgemv(std::string routine_name, FblasTranspose transposed, unsigned int N, unsigned int M, double alpha, cl::Buffer A,
                             unsigned int lda, cl::Buffer x, int incx, double beta, cl::Buffer y, int incy, std::vector<cl::Event> *events_wait_list, cl::Event *event)
{
    FBLASEnvironment::gemv<double>(routine_name,transposed,N,M,alpha,A,lda,x,incx,beta,y,incy,events_wait_list,event);

}

/*------------------
 * TRMV
 * ------------------*/

template <typename T>
void FBLASEnvironment::trmv(std::string routine_name, FblasUpLo uplo, FblasTranspose trans,FblasDiag diag, unsigned int N, cl::Buffer A,
                            unsigned int lda, cl::Buffer x, int incx, std::vector<cl::Event> * events_wait_list, cl::Event * event )
{
    auto it= routines_.find(routine_name);
    if(it == routines_.end())
    {
        std::cerr << "There not exist a routine with this given name: "<<routine_name  << std::endl;
        return;
    }
    Routine r=it->second;

    FblasOrder ord=r.order;

    //checks
    CHECK_TRANS(trans,r);
    CHECK_INCX(incx,r);
    CHECK_UPLO(uplo,r);

    //TRMV will be realized using GEMV: x will be copied in a new buffer z
    // and the computation x=Az will be performed
    //create the buffer and copy y
    cl::Buffer z(FBLASEnvironment::get_context(), CL_MEM_READ_ONLY, N * abs(incx)*sizeof(T));
    r.queues[0].enqueueCopyBuffer(x,z,0,0,N*abs(incx)*sizeof(T));
    r.queues[0].finish();

    //so in the following with x we indicate z, and with y we refer to x
    unsigned int x_repetitions= x_repetitions=ceil((float)(N)/r.tile_n_size);
    unsigned int y_repetitions=0;
    int order=(ord==FBLAS_ROW_MAJOR)?1:0;
    unsigned int x_size=N;
    unsigned int y_size=N;
    unsigned int tile_x_size=(trans==FBLAS_TRANSPOSED)? r.tile_n_size:r.tile_m_size;
    unsigned int tile_y_size=(trans==FBLAS_TRANSPOSED)? r.tile_m_size:r.tile_n_size;
    //unsigned int lda=(r.lda==0)?M:r.lda;

    //Set kernel arguments, according to the routine characteristics
    //gemv
    T fone=1.0;
    T zero=0;
    r.kernels[0].setArg(0, sizeof(int),&order);
    r.kernels[0].setArg(1, sizeof(int),&N);
    r.kernels[0].setArg(2, sizeof(int),&N);
    r.kernels[0].setArg(3, sizeof(T),&fone);
    r.kernels[0].setArg(4, sizeof(T),&zero);
    //matrix reader
    r.kernels[1].setArg(0, sizeof(cl_mem),&A);
    r.kernels[1].setArg(1, sizeof(int),&N);
    r.kernels[1].setArg(2, sizeof(int),&lda);

    //read x and y
    r.kernels[2].setArg(0,sizeof(cl_mem), &z);
    r.kernels[2].setArg(1, sizeof(unsigned int),&x_size);
    r.kernels[2].setArg(2, sizeof(unsigned int),&tile_x_size);
    r.kernels[2].setArg(3, sizeof(unsigned int),&x_repetitions);

    r.kernels[3].setArg(0,sizeof(cl_mem), &x);
    r.kernels[3].setArg(1, sizeof(unsigned int),&y_size);
    r.kernels[3].setArg(2, sizeof(unsigned int),&tile_y_size);
    r.kernels[3].setArg(3, sizeof(unsigned int),&y_repetitions);


    r.kernels[4].setArg(0,sizeof(cl_mem), &x);
    r.kernels[4].setArg(1,sizeof(int),&y_size);  //attention to this
    r.kernels[4].setArg(2,sizeof(int),&tile_y_size);

    //launch kernels: if the routine is non-blocking (i.e. the event parameter is not null)
    //we create the corresponding event when we launch the sink kernel (the last one)

    for(int i=0;i<r.kernels.size()-1;i++)
        r.queues[i].enqueueTask(r.kernels[i],events_wait_list);

    //launch the last one
    r.queues[r.kernels.size()-1].enqueueTask(r.kernels[r.kernels.size()-1],events_wait_list,event);

    if(!event)
        for(int i=0;i<r.kernels.size();i++)
            r.queues[i].finish();

    //since we are working with C++ Opencl, z will be (should be) released when its ref counts arrive to zero
}



void FBLASEnvironment::strmv(std::string routine_name, FblasUpLo uplo, FblasTranspose trans,FblasDiag diag, unsigned int N, cl::Buffer A,
                              unsigned int lda, cl::Buffer x, int incx, std::vector<cl::Event> * events_wait_list, cl::Event * event)
{
    FBLASEnvironment::trmv<float>(routine_name,uplo,trans,diag,N,A,lda,x,incx,events_wait_list,event);
}

void FBLASEnvironment::dtrmv(std::string routine_name, FblasUpLo uplo, FblasTranspose trans,FblasDiag diag, unsigned int N, cl::Buffer A,
                              unsigned int lda, cl::Buffer x, int incx, std::vector<cl::Event> * events_wait_list, cl::Event * event)
{
    FBLASEnvironment::trmv<double>(routine_name,uplo,trans,diag,N,A,lda,x,incx,events_wait_list,event);
}


/*---------------------
 * SYMV
 * ---------------------*/
template <typename T>
void FBLASEnvironment::symv(std::string routine_name, FblasUpLo uplo, unsigned int N, T alpha, cl::Buffer A, unsigned int lda,
           cl::Buffer x, int incx, T beta, cl::Buffer y,  int incy, std::vector<cl::Event> * events_wait_list, cl::Event * event)
{
    auto it= routines_.find(routine_name);
    if(it == routines_.end())
    {
        std::cerr << "There not exist a routine with this given name: "<<routine_name  << std::endl;
        return;
    }
    Routine r=it->second;

    FblasOrder ord=r.order;

    CHECK_UPLO(uplo,r);
    CHECK_INCX(incx,r);
    CHECK_INCY(incy,r);

    unsigned int x_repetitions=1;
    unsigned int y_repetitions=(beta==0)?0:1;
    int order=(ord==FBLAS_ROW_MAJOR)?1:0;
    unsigned int x_size=N;
    unsigned int y_size=N;
    unsigned int tile_y_size=r.tile_n_size;
    unsigned int tile_x_size=r.tile_n_size;

    x_repetitions=ceil((float)(N)/r.tile_n_size);

    //Set kernel arguments, according to the routine characteristics
    //gemv

    r.kernels[0].setArg(0, sizeof(int),&order);
    r.kernels[0].setArg(1, sizeof(int),&N);
    r.kernels[0].setArg(2, sizeof(int),&N);
    r.kernels[0].setArg(3, sizeof(T),&alpha);
    r.kernels[0].setArg(4, sizeof(T),&beta);
    //matrix reader
    r.kernels[1].setArg(0, sizeof(cl_mem),&A);
    r.kernels[1].setArg(1, sizeof(int),&N);
    r.kernels[1].setArg(2, sizeof(int),&lda);

    //read x and y
    r.kernels[2].setArg(0,sizeof(cl_mem), &x);
    r.kernels[2].setArg(1, sizeof(unsigned int),&x_size);
    r.kernels[2].setArg(2, sizeof(unsigned int),&tile_x_size);
    r.kernels[2].setArg(3, sizeof(unsigned int),&x_repetitions);

    r.kernels[3].setArg(0,sizeof(cl_mem), &y);
    r.kernels[3].setArg(1, sizeof(unsigned int),&y_size);
    r.kernels[3].setArg(2, sizeof(unsigned int),&tile_y_size);
    r.kernels[3].setArg(3, sizeof(unsigned int),&y_repetitions);


    r.kernels[4].setArg(0,sizeof(cl_mem), &y);
    r.kernels[4].setArg(1,sizeof(int),&y_size);  //attention to this
    r.kernels[4].setArg(2,sizeof(int),&tile_y_size);

    //launch kernels: if the routine is non-blocking (i.e. the event parameter is not null)
    //we create the corresponding event when we launch the sink kernel (the last one)

    for(int i=0;i<r.kernels.size()-1;i++)
        r.queues[i].enqueueTask(r.kernels[i],events_wait_list);

    //launch the last one
    r.queues[r.kernels.size()-1].enqueueTask(r.kernels[r.kernels.size()-1],events_wait_list,event);

    if(!event)
        for(int i=0;i<r.kernels.size();i++)
            r.queues[i].finish();
}

void FBLASEnvironment::ssymv(std::string routine_name, FblasUpLo uplo, unsigned int N, float alpha, cl::Buffer A, unsigned int lda,
                             cl::Buffer x, int incx, float beta, cl::Buffer y, int incy, std::vector<cl::Event> *events_wait_list, cl::Event *event)
{
    FBLASEnvironment::symv<float>(routine_name,uplo,N,alpha,A,lda,x,incx,beta,y,incy,events_wait_list,event);
}

void FBLASEnvironment::dsymv(std::string routine_name, FblasUpLo uplo, unsigned int N, double alpha, cl::Buffer A, unsigned int lda,
                             cl::Buffer x, int incx, double beta, cl::Buffer y, int incy, std::vector<cl::Event> *events_wait_list, cl::Event *event)
{
    FBLASEnvironment::symv<double>(routine_name,uplo,N,alpha,A,lda,x,incx,beta,y,incy,events_wait_list,event);
}


/*-----------------------
 * GER
 *-----------------------*/

template <typename T>
void FBLASEnvironment::ger(std::string routine_name, unsigned int N, unsigned int M, T alpha, cl::Buffer x, int incx,  cl::Buffer y,
         int incy,  cl::Buffer A, unsigned int lda, std::vector<cl::Event> * events_wait_list, cl::Event * event)
{
    auto it= routines_.find(routine_name);
    if(it == routines_.end())
    {
        std::cerr << "There not exist a routine with this given name: "<<routine_name  << std::endl;
        return;
    }
    Routine r=it->second;

    FblasOrder ord=r.order;

    unsigned int x_repetitions=1;
    unsigned int y_repetitions=1;

    unsigned int tile_y_size= r.tile_m_size;
    unsigned int tile_x_size= r.tile_n_size;

    if(ord==FBLAS_ROW_MAJOR) //in this case x must be repeated N/tile_N times
        y_repetitions=ceil((float)(N)/r.tile_n_size);


    //Set kernel arguments, according to the routine characteristics
    //ger

    r.kernels[0].setArg(0, sizeof(T),&alpha);
    r.kernels[0].setArg(1, sizeof(int),&N);
    r.kernels[0].setArg(2, sizeof(int),&M);

    //matrix reader
    r.kernels[1].setArg(0, sizeof(cl_mem),&A);
    r.kernels[1].setArg(1, sizeof(int),&N);
    r.kernels[1].setArg(2, sizeof(int),&M);
    r.kernels[1].setArg(3, sizeof(int),&lda);


    //read x and y
    r.kernels[2].setArg(0,sizeof(cl_mem), &x);
    r.kernels[2].setArg(1, sizeof(unsigned int),&N);
    r.kernels[2].setArg(2, sizeof(unsigned int),&tile_x_size);
    r.kernels[2].setArg(3, sizeof(unsigned int),&x_repetitions);

    r.kernels[3].setArg(0,sizeof(cl_mem), &y);
    r.kernels[3].setArg(1, sizeof(unsigned int),&M);
    r.kernels[3].setArg(2, sizeof(unsigned int),&tile_y_size);
    r.kernels[3].setArg(3, sizeof(unsigned int),&y_repetitions);

    //writer
    r.kernels[4].setArg(0,sizeof(cl_mem), &A);
    r.kernels[4].setArg(1, sizeof(int),&N);
    r.kernels[4].setArg(2, sizeof(int),&M);
    r.kernels[4].setArg(3, sizeof(int),&lda);

    //launch kernels: if the routine is non-blocking (i.e. the event parameter is not null)
    //we create the corresponding event when we launch the sink kernel (the last one)

    for(int i=0;i<r.kernels.size()-1;i++)
        r.queues[i].enqueueTask(r.kernels[i],events_wait_list);

    //launch the last one
    r.queues[r.kernels.size()-1].enqueueTask(r.kernels[r.kernels.size()-1],events_wait_list,event);

    if(!event)
        for(int i=0;i<r.kernels.size();i++)
            r.queues[i].finish();
}

void FBLASEnvironment::sger(std::string routine_name, unsigned int N, unsigned int M, float alpha, cl::Buffer x, int incx,
          cl::Buffer y, int incy,  cl::Buffer A, unsigned int lda, std::vector<cl::Event> * events_wait_list, cl::Event * event)
{
    FBLASEnvironment::ger<float>(routine_name,N,M,alpha,x,incx,y,incy,A,lda,events_wait_list,event);

}

void FBLASEnvironment::dger(std::string routine_name, unsigned int N, unsigned int M, double alpha, cl::Buffer x, int incx,
          cl::Buffer y, int incy,  cl::Buffer A, unsigned int lda, std::vector<cl::Event> * events_wait_list, cl::Event * event)
{
    FBLASEnvironment::ger<double>(routine_name,N,M,alpha,x,incx,y,incy,A,lda,events_wait_list,event);

}


/*----------
 *  SYR
 *--------------*/

template <typename T>
void FBLASEnvironment::syr(std::string routine_name, FblasUpLo uplo, unsigned int N, T alpha, cl::Buffer x, int incx, cl::Buffer A,
                            unsigned int lda, std::vector<cl::Event> *events_wait_list, cl::Event *event)
{
    auto it= routines_.find(routine_name);
    if(it == routines_.end())
    {
        std::cerr << "There not exist a routine with this given name: "<<routine_name  << std::endl;
        return;
    }
    Routine r=it->second;

    FblasOrder ord=r.order;
    CHECK_INCX(incx,r);
    CHECK_UPLO(uplo,r);

    unsigned int x_repetitions=1;
    unsigned int tile_x_size=r.tile_n_size;

    //Set kernel arguments, according to the routine characteristics
    //ger

    r.kernels[0].setArg(0, sizeof(T),&alpha);
    r.kernels[0].setArg(1, sizeof(int),&N);

    //matrix reader
    r.kernels[1].setArg(0, sizeof(cl_mem),&A);
    r.kernels[1].setArg(1, sizeof(int),&N);
    r.kernels[1].setArg(2, sizeof(int),&lda);


    //read x and x transposed
    r.kernels[2].setArg(0,sizeof(cl_mem), &x);
    r.kernels[2].setArg(1, sizeof(unsigned int),&N);
    r.kernels[2].setArg(2, sizeof(unsigned int),&tile_x_size);
    r.kernels[2].setArg(3, sizeof(unsigned int),&x_repetitions);

    r.kernels[3].setArg(0,sizeof(cl_mem), &x);
    r.kernels[3].setArg(1, sizeof(unsigned int),&N);

    //writer
    r.kernels[4].setArg(0,sizeof(cl_mem), &A);
    r.kernels[4].setArg(1, sizeof(int),&N);
    r.kernels[4].setArg(2, sizeof(int),&lda);

    //launch kernels: if the routine is non-blocking (i.e. the event parameter is not null)
    //we create the corresponding event when we launch the sink kernel (the last one)

    for(int i=0;i<r.kernels.size()-1;i++)
        r.queues[i].enqueueTask(r.kernels[i],events_wait_list);

    //launch the last one
    r.queues[r.kernels.size()-1].enqueueTask(r.kernels[r.kernels.size()-1],events_wait_list,event);

    if(!event)
        for(int i=0;i<r.kernels.size();i++)
            r.queues[i].finish();
}

void FBLASEnvironment::ssyr(std::string routine_name, FblasUpLo uplo, unsigned int N, float alpha, cl::Buffer x, int incx, cl::Buffer A,
                            unsigned int lda, std::vector<cl::Event> *events_wait_list, cl::Event *event)
{
   FBLASEnvironment::syr<float>(routine_name,uplo, N,alpha,x,incx,A,lda,events_wait_list,event);
}

void FBLASEnvironment::dsyr(std::string routine_name, FblasUpLo uplo, unsigned int N, double alpha, cl::Buffer x, int incx, cl::Buffer A,
                            unsigned int lda, std::vector<cl::Event> *events_wait_list, cl::Event *event)
{
   FBLASEnvironment::syr<double>(routine_name,uplo, N,alpha,x,incx,A,lda,events_wait_list,event);
}


/*-------
 *  SYR2
 *--------*/

template <typename T>
void FBLASEnvironment::syr2(std::string routine_name, FblasUpLo uplo, unsigned int N, T alpha, cl::Buffer x, int incx, cl::Buffer y, int incy,
                            cl::Buffer A, unsigned int lda, std::vector<cl::Event> *events_wait_list, cl::Event *event)
{
    auto it= routines_.find(routine_name);
    if(it == routines_.end())
    {
        std::cerr << "There not exist a routine with this given name: "<<routine_name  << std::endl;
        return;
    }
    Routine r=it->second;

    FblasOrder ord=r.order;
    CHECK_INCX(incx,r);
    CHECK_INCY(incy,r);
    CHECK_UPLO(uplo,r);

    unsigned int x_repetitions=1;
    unsigned int tile_x_size=r.tile_n_size;

    //Set kernel arguments, according to the routine characteristics
    //ger

    r.kernels[0].setArg(0, sizeof(T),&alpha);
    r.kernels[0].setArg(1, sizeof(int),&N);

    //matrix reader
    r.kernels[1].setArg(0, sizeof(cl_mem),&A);
    r.kernels[1].setArg(1, sizeof(int),&N);
    r.kernels[1].setArg(2, sizeof(int),&lda);


    //read x and x transposed
    r.kernels[2].setArg(0,sizeof(cl_mem), &x);
    r.kernels[2].setArg(1, sizeof(unsigned int),&N);
    r.kernels[2].setArg(2, sizeof(unsigned int),&tile_x_size);
    r.kernels[2].setArg(3, sizeof(unsigned int),&x_repetitions);

    r.kernels[3].setArg(0,sizeof(cl_mem), &x);
    r.kernels[3].setArg(1, sizeof(unsigned int),&N);

    //read y and y transposed
    r.kernels[4].setArg(0,sizeof(cl_mem), &y);
    r.kernels[4].setArg(1, sizeof(unsigned int),&N);
    r.kernels[4].setArg(2, sizeof(unsigned int),&tile_x_size);
    r.kernels[4].setArg(3, sizeof(unsigned int),&x_repetitions);

    r.kernels[5].setArg(0,sizeof(cl_mem), &y);
    r.kernels[5].setArg(1, sizeof(unsigned int),&N);

    //writer
    r.kernels[6].setArg(0,sizeof(cl_mem), &A);
    r.kernels[6].setArg(1, sizeof(int),&N);
    r.kernels[6].setArg(2, sizeof(int),&lda);

    //launch kernels: if the routine is non-blocking (i.e. the event parameter is not null)
    //we create the corresponding event when we launch the sink kernel (the last one)

    for(int i=0;i<r.kernels.size()-1;i++)
        r.queues[i].enqueueTask(r.kernels[i],events_wait_list);

    //launch the last one
    r.queues[r.kernels.size()-1].enqueueTask(r.kernels[r.kernels.size()-1],events_wait_list,event);

    if(!event)
        for(int i=0;i<r.kernels.size();i++)
            r.queues[i].finish();
}

void FBLASEnvironment::ssyr2(std::string routine_name, FblasUpLo uplo, unsigned int N, float alpha, cl::Buffer x, int incx, cl::Buffer y, int incy,
                            cl::Buffer A, unsigned int lda, std::vector<cl::Event> *events_wait_list, cl::Event *event)
{
   FBLASEnvironment::syr2<float>(routine_name,uplo, N,alpha,x,incx,y,incy,A,lda,events_wait_list,event);
}

void FBLASEnvironment::dsyr2(std::string routine_name, FblasUpLo uplo, unsigned int N, double alpha, cl::Buffer x, int incx, cl::Buffer y, int incy,
                             cl::Buffer A, unsigned int lda, std::vector<cl::Event> *events_wait_list, cl::Event *event)
{
   FBLASEnvironment::syr2<double>(routine_name,uplo, N,alpha,x,incx,y,incy,A,lda,events_wait_list,event);
}



/*------------------
 * TRSV
 *------------------------*/


template <typename T>
void FBLASEnvironment::trsv(std::string routine_name,FblasUpLo ul, FblasTranspose t, unsigned int N, cl::Buffer A, unsigned int lda,
                            cl::Buffer x, int incx, std::vector<cl::Event> * events_wait_list, cl::Event * event)
{
    auto it= routines_.find(routine_name);
    if(it == routines_.end())
    {
        std::cerr << "There not exist a routine with this given name: "<<routine_name  << std::endl;
        return;
    }
    Routine r=it->second;

    FblasOrder ord=r.order;
    FblasTranspose trans=r.transA;
    FblasUpLo uplo=r.uplo;
    unsigned int tile_x_size=r.tile_n_size;

    //Set kernel arguments, according to the routine characteristics


    r.kernels[0].setArg(0, sizeof(int),&N);

    //matrix reader
    r.kernels[1].setArg(0, sizeof(cl_mem),&A);
    r.kernels[1].setArg(1, sizeof(int),&N);
    r.kernels[1].setArg(2, sizeof(int),&lda);


    //read/write x
    r.kernels[2].setArg(0,sizeof(cl_mem), &x);
    r.kernels[2].setArg(1, sizeof(unsigned int),&N);

    //launch kernels: if the routine is non-blocking (i.e. the event parameter is not null)
    //we create the corresponding event when we launch the sink kernel (the last one)

    for(int i=0;i<r.kernels.size()-1;i++)
        r.queues[i].enqueueTask(r.kernels[i],events_wait_list);

    //launch the last one
    r.queues[r.kernels.size()-1].enqueueTask(r.kernels[r.kernels.size()-1],events_wait_list,event);

    if(!event)
        for(int i=0;i<r.kernels.size();i++)
            r.queues[i].finish();
}

void FBLASEnvironment::strsv(std::string routine_name, FblasUpLo uplo, FblasTranspose trans, unsigned int N, cl::Buffer A,
                             unsigned int lda, cl::Buffer x, int incx, std::vector<cl::Event> *events_wait_list, cl::Event *event)
{
    FBLASEnvironment::trsv<float>(routine_name,uplo,trans,N,A,lda,x,incx,events_wait_list,event);
}


void FBLASEnvironment::dtrsv(std::string routine_name, FblasUpLo uplo, FblasTranspose trans, unsigned int N, cl::Buffer A,
                             unsigned int lda, cl::Buffer x, int incx, std::vector<cl::Event> *events_wait_list, cl::Event *event)
{
    FBLASEnvironment::trsv<double>(routine_name,uplo,trans,N,A,lda,x,incx,events_wait_list,event);
}

#endif // FBLAS_ENVIRONMENT_LEVEL2_HPP
