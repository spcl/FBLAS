/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2019 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    Host Api Implementation - Level 3 Routines
*/


#ifndef FBLAS_ENVIRONMENT_LEVEL3_HPP
#define FBLAS_ENVIRONMENT_LEVEL3_HPP

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <math.h>
#include <rapidjson/document.h>
#include "../fblas_environment.hpp"
#include "../utils/ocl_utils.hpp"
#define CHECK_TRANSA(TA,R) { if(TA != R.transA){std::cerr << "Wrong \"transa\" parameter for routine: "<<R.user_name  << std::endl;return;}}
#define CHECK_TRANSB(TB,R) { if(TB != R.transB){std::cerr << "Wrong \"transb\" parameter for routine: "<<R.user_name  << std::endl;return;}}
#define CHECK_UPLO(U,R) { if(U != R.uplo) {std::cerr << "Wrong \"uplo\" parameter for routine: "<<R.user_name << std::endl;  return; }}


/*--------------
 * GEMM
 *---------------*/

template <typename T>
void FBLASEnvironment::gemm(std::string routine_name, FblasTranspose transA, FblasTranspose transB, unsigned int N, unsigned int M, unsigned int K,
           T alpha, cl::Buffer A, unsigned int lda, cl::Buffer B, unsigned int ldb, T beta, cl::Buffer C, unsigned int ldc,
           std::vector<cl::Event> * events_wait_list, cl::Event * event )
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
    CHECK_TRANSA(transA,r);
    CHECK_TRANSB(transB,r);

    //gemm is composed by 4 kernels
    r.kernels[0].setArg(0,sizeof(unsigned int),&N);
    r.kernels[0].setArg(1,sizeof(unsigned int),&M);
    r.kernels[0].setArg(2,sizeof(unsigned int),&K);
    r.kernels[0].setArg(3,sizeof(T),&alpha);
    r.kernels[1].setArg(0,sizeof(cl_mem),&A);
    r.kernels[1].setArg(1,sizeof(unsigned int),&N);
    r.kernels[1].setArg(2,sizeof(unsigned int),&K);
    r.kernels[1].setArg(3,sizeof(unsigned int),&M);
    r.kernels[1].setArg(4,sizeof(unsigned int),&lda);
    r.kernels[2].setArg(0,sizeof(cl_mem),&B);
    r.kernels[2].setArg(1,sizeof(unsigned int),&N);
    r.kernels[2].setArg(2,sizeof(unsigned int),&K);
    r.kernels[2].setArg(3,sizeof(unsigned int),&M);
    r.kernels[2].setArg(4,sizeof(unsigned int),&ldb);
    r.kernels[3].setArg(0,sizeof(cl_mem),&C);
    r.kernels[3].setArg(1,sizeof(T),&beta);
    r.kernels[3].setArg(2,sizeof(unsigned int),&N);
    r.kernels[3].setArg(3,sizeof(unsigned int),&M);
    r.kernels[3].setArg(4,sizeof(unsigned int),&ldc);

    //launch kernels: if the routine is non-blocking (i.e. the event parameter is not null)
    //we create the corresponding event
    for(int i=0;i<r.kernels.size()-1;i++)
    {
        if(i!=0 || (!r.systolic || !this->running_)) //do not start again computational systolic kernel
            r.queues[i].enqueueTask(r.kernels[i],events_wait_list);
        else
            std::cout << "Non faccio partire il kernel "<<i<<std::endl;

    }

    if(r.systolic)
        this->running_=true;

    //launch the last one
    r.queues[r.kernels.size()-1].enqueueTask(r.kernels[r.kernels.size()-1],events_wait_list,event);

    //if this is systolic, we don't have to wait for the computational module
    //TODO: handle this properly with asyncronous fblas call
    if(!event)
        for(int i=0;i<r.kernels.size();i++)
        {
            //if(!r.systolic || i!=0)
             r.queues[i].finish();
        }
}


void FBLASEnvironment::sgemm(std::string routine_name, FblasTranspose transA, FblasTranspose transB, unsigned int N, unsigned int M, unsigned int K,
           float alpha, cl::Buffer A, unsigned int lda, cl::Buffer B, unsigned int ldb, float beta, cl::Buffer C, unsigned int ldc,
           std::vector<cl::Event> * events_wait_list, cl::Event * event )
{
    FBLASEnvironment::gemm<float>(routine_name,transA,transB,N,M,K,alpha,A,lda,B,ldb,beta,C,ldc,events_wait_list,event);
}

void FBLASEnvironment::dgemm(std::string routine_name, FblasTranspose transA, FblasTranspose transB, unsigned int N, unsigned int M, unsigned int K,
           double alpha, cl::Buffer A, unsigned int lda, cl::Buffer B, unsigned int ldb, double beta, cl::Buffer C, unsigned int ldc,
           std::vector<cl::Event> * events_wait_list, cl::Event * event )
{
    FBLASEnvironment::gemm<double>(routine_name,transA,transB,N,M,K,alpha,A,lda,B,ldb,beta,C,ldc,events_wait_list,event);
}


/*----------------
 * SYRK
 * --------------*/
template <typename T>
void FBLASEnvironment::syrk(std::string routine_name, FblasUpLo uplo, FblasTranspose trans, unsigned int N, unsigned int K, T alpha,
                             cl::Buffer A, unsigned int lda, T beta, cl::Buffer C, unsigned int ldc,
                             std::vector<cl::Event> *events_wait_list, cl::Event *event)
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
    CHECK_TRANSA(trans,r);
    CHECK_UPLO(uplo,r);

    //syrk is composed by 4 kernels
    //we have to specify as argument if C is lower or not
    int lower=(uplo==FBLAS_LOWER)?1:0;
    //main kernel
    r.kernels[0].setArg(0,sizeof(T),&alpha);
    r.kernels[0].setArg(1,sizeof(unsigned int),&N);
    r.kernels[0].setArg(2,sizeof(unsigned int),&K);
    r.kernels[0].setArg(3,sizeof(unsigned int),&lower);

    //read A
    r.kernels[1].setArg(0,sizeof(cl_mem),&A);
    r.kernels[1].setArg(1,sizeof(unsigned int),&N);
    r.kernels[1].setArg(2,sizeof(unsigned int),&K);
    r.kernels[1].setArg(3,sizeof(unsigned int),&lda);
    r.kernels[1].setArg(4,sizeof(unsigned int),&lower);


    //read A2
    r.kernels[2].setArg(0,sizeof(cl_mem),&A);
    r.kernels[2].setArg(1,sizeof(unsigned int),&N);
    r.kernels[2].setArg(2,sizeof(unsigned int),&K);
    r.kernels[2].setArg(3,sizeof(unsigned int),&lda);
    r.kernels[2].setArg(4,sizeof(unsigned int),&lower);


    //writer
    r.kernels[3].setArg(0,sizeof(cl_mem),&C);
    r.kernels[3].setArg(1,sizeof(T),&beta);
    r.kernels[3].setArg(2,sizeof(int),&N);
    r.kernels[3].setArg(3,sizeof(int),&ldc);


    //launch kernels: if the routine is non-blocking (i.e. the event parameter is not null)
    //we create the corresponding event
    for(int i=0;i<r.kernels.size()-1;i++)
        r.queues[i].enqueueTask(r.kernels[i],events_wait_list);

    //launch the last one
    r.queues[r.kernels.size()-1].enqueueTask(r.kernels[r.kernels.size()-1],events_wait_list,event);

    if(!event)
        for(int i=0;i<r.kernels.size();i++)
            r.queues[i].finish();
}

void FBLASEnvironment::ssyrk(std::string routine_name, FblasUpLo uplo, FblasTranspose trans, unsigned int N, unsigned int K, float alpha,
                             cl::Buffer A, unsigned int lda, float beta, cl::Buffer C, unsigned int ldc,
                             std::vector<cl::Event> *events_wait_list, cl::Event *event)
{
    FBLASEnvironment::syrk<float>(routine_name,uplo,trans,N, K, alpha,A,lda,beta,C,ldc,events_wait_list,event);
}

void FBLASEnvironment::dsyrk(std::string routine_name, FblasUpLo uplo, FblasTranspose trans, unsigned int N, unsigned int K, double alpha,
                             cl::Buffer A, unsigned int lda, double beta, cl::Buffer C, unsigned int ldc,
                             std::vector<cl::Event> *events_wait_list, cl::Event *event)
{
    FBLASEnvironment::syrk<double>(routine_name,uplo,trans,N, K, alpha,A,lda,beta,C,ldc,events_wait_list,event);
}

/*---------------
 * SYR2K
 *-------------------*/

template <typename T>
void FBLASEnvironment::syr2k(std::string routine_name, FblasUpLo uplo, FblasTranspose trans, unsigned int N, unsigned int K,
                              T alpha, cl::Buffer A, unsigned int lda, cl::Buffer B, unsigned int ldb, T beta, cl::Buffer C, unsigned int ldc,
                              std::vector<cl::Event> * events_wait_list, cl::Event * event)
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
    CHECK_TRANSA(trans,r);
    CHECK_UPLO(uplo,r);

    //syrk is composed by 6 kernels
    //we have to specify as argument if C is lower or not
    int lower=(uplo==FBLAS_LOWER)?1:0;
    //main kernel
    r.kernels[0].setArg(0,sizeof(T),&alpha);
    r.kernels[0].setArg(1,sizeof(unsigned int),&N);
    r.kernels[0].setArg(2,sizeof(unsigned int),&K);
    r.kernels[0].setArg(3,sizeof(unsigned int),&lower);

    //read A
    r.kernels[1].setArg(0,sizeof(cl_mem),&A);
    r.kernels[1].setArg(1,sizeof(unsigned int),&N);
    r.kernels[1].setArg(2,sizeof(unsigned int),&K);
    r.kernels[1].setArg(3,sizeof(unsigned int),&lda);
    r.kernels[1].setArg(4,sizeof(unsigned int),&lower);


    //read A2
    r.kernels[2].setArg(0,sizeof(cl_mem),&A);
    r.kernels[2].setArg(1,sizeof(unsigned int),&N);
    r.kernels[2].setArg(2,sizeof(unsigned int),&K);
    r.kernels[2].setArg(3,sizeof(unsigned int),&lda);
    r.kernels[2].setArg(4,sizeof(unsigned int),&lower);

    //read B
    r.kernels[3].setArg(0,sizeof(cl_mem),&B);
    r.kernels[3].setArg(1,sizeof(int),&N);
    r.kernels[3].setArg(2,sizeof(int),&K);
    r.kernels[3].setArg(3,sizeof(int),&ldb);
    r.kernels[3].setArg(4,sizeof(int),&lower);
    //READ B2
    r.kernels[4].setArg(0,sizeof(cl_mem),&B);
    r.kernels[4].setArg(1,sizeof(int),&N);
    r.kernels[4].setArg(2,sizeof(int),&K);
    r.kernels[4].setArg(3,sizeof(int),&lda);
    r.kernels[4].setArg(4,sizeof(int),&lower);


    //writer
    r.kernels[5].setArg(0,sizeof(cl_mem),&C);
    r.kernels[5].setArg(1,sizeof(T),&beta);
    r.kernels[5].setArg(2,sizeof(int),&N);
    r.kernels[5].setArg(3,sizeof(int),&ldc);


    //launch kernels: if the routine is non-blocking (i.e. the event parameter is not null)
    //we create the corresponding event
    for(int i=0;i<r.kernels.size()-1;i++)
        r.queues[i].enqueueTask(r.kernels[i],events_wait_list);

    //launch the last one
    r.queues[r.kernels.size()-1].enqueueTask(r.kernels[r.kernels.size()-1],events_wait_list,event);

    if(!event)
        for(int i=0;i<r.kernels.size();i++)
            r.queues[i].finish();
}

void FBLASEnvironment::ssyr2k(std::string routine_name, FblasUpLo uplo, FblasTranspose trans, unsigned int N, unsigned int K,
                              float alpha, cl::Buffer A, unsigned int lda, cl::Buffer B, unsigned int ldb, float beta,
                              cl::Buffer C, unsigned int ldc, std::vector<cl::Event> *events_wait_list, cl::Event *event)
{
    FBLASEnvironment::syr2k<float>(routine_name,uplo,trans,N,K,alpha,A,lda,B,ldb,beta,C,ldc,events_wait_list,event);
}

void FBLASEnvironment::dsyr2k(std::string routine_name, FblasUpLo uplo, FblasTranspose trans, unsigned int N, unsigned int K,
                              double alpha, cl::Buffer A, unsigned int lda, cl::Buffer B, unsigned int ldb, double beta,
                              cl::Buffer C, unsigned int ldc, std::vector<cl::Event> *events_wait_list, cl::Event *event)
{
    FBLASEnvironment::syr2k<double>(routine_name,uplo,trans,N,K,alpha,A,lda,B,ldb,beta,C,ldc,events_wait_list,event);
}


/*------------
 * TRSM
 * -----------*/

template <typename T>
void FBLASEnvironment::trsm(std::string routine_name,FblasSide side, FblasTranspose trans, FblasUpLo uplo, unsigned int N, unsigned int M,
          T alpha, cl::Buffer A, unsigned int lda, cl::Buffer B, unsigned int ldb, std::vector<cl::Event> * events_wait_list, cl::Event * event)
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
    if(side!= r.side){
        std::cerr << "Wrong \"side\" parameter for routine: "<<routine_name  << std::endl;
        return;
    }

    if(trans != r.transA)
    {
        std::cerr << "Wrong \"transa\" parameter for routine: "<<routine_name  << std::endl;
        return;
    }
    if(uplo != r.uplo)
    {
        std::cerr << "Wrong \"uplo\" parameter for routine: "<<routine_name  << std::endl;
        return;
    }


    //Set kernel arguments, according to the routine characteristics

    r.kernels[0].setArg(0,sizeof(int),&N);
    r.kernels[0].setArg(1,sizeof(int),&M);
    r.kernels[0].setArg(2,sizeof(T),&alpha);
    r.kernels[0].setArg(3,sizeof(cl_mem),&A);
    r.kernels[0].setArg(4,sizeof(unsigned int),&lda);
    r.kernels[0].setArg(5,sizeof(cl_mem),&B);
    r.kernels[0].setArg(6,sizeof(unsigned int),&M);

    //launch kernels: if the routine is non-blocking (i.e. the event parameter is not null)
    //we create the corresponding event

    r.queues[0].enqueueTask(r.kernels[0],events_wait_list,event);

    if(!event) //blocking call
        r.queues[0].finish();
}


void FBLASEnvironment::strsm(std::string routine_name,FblasSide side, FblasTranspose transa, FblasUpLo uplo, unsigned int N, unsigned int M,
          float alpha, cl::Buffer A, unsigned int lda, cl::Buffer B, unsigned int ldb, std::vector<cl::Event> * events_wait_list, cl::Event * event)
{
    FBLASEnvironment::trsm<float>(routine_name,side,transa,uplo,N,M,alpha,A,lda,B,ldb,events_wait_list,event);
}

void FBLASEnvironment::dtrsm(std::string routine_name,FblasSide side, FblasTranspose transa, FblasUpLo uplo, unsigned int N, unsigned int M,
          double alpha, cl::Buffer A, unsigned int lda, cl::Buffer B, unsigned int ldb, std::vector<cl::Event> * events_wait_list, cl::Event * event)
{
    FBLASEnvironment::trsm<double>(routine_name,side,transa,uplo,N,M,alpha,A,lda,B,ldb,events_wait_list,event);
}




#endif // FBLAS_ENVIRONMENT_LEVEL3_HPP


