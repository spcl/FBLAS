/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2019 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    Host Api Implementation - Level 1 Routines
*/
#ifndef FBLAS_ENVIRONMENT_LEVEL1_HPP
#define FBLAS_ENVIRONMENT_LEVEL1_HPP

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <math.h>
#include <rapidjson/document.h>
#include "../fblas_environment.hpp"
#include "../utils/ocl_utils.hpp"

//#include "../../utils/includes/utils.hpp"


/*-----------
 * SWAP
 * ------------*/



template <typename T>
void FBLASEnvironment::swap(std::string routine_name,  unsigned int N, cl::Buffer x, int incx,  cl::Buffer y, int incy,
                            std::vector<cl::Event> * events_wait_list, cl::Event * event)
{
    //Set kernel args
    if(routines_.find(routine_name)==routines_.end())
    {
        std::cerr << "Routine "<<routine_name<<" not present" <<std::endl;
        return;
    }
    Routine r=routines_[routine_name];

    CHECK_INCX(incx,r);
    CHECK_INCY(incy,r);

    unsigned int width=r.width;
    unsigned int one=1;
    //set args: readers inject data with padding size equal to W

    r.kernels[0].setArg(0, sizeof(unsigned int),&N);
    r.kernels[1].setArg(0,sizeof(cl_mem), &x);
    r.kernels[1].setArg(1,sizeof(int),&N);
    r.kernels[1].setArg(2, sizeof(unsigned int),&width);
    r.kernels[1].setArg(3, sizeof(unsigned int),&one);

    r.kernels[2].setArg(0,sizeof(cl_mem), &y);
    r.kernels[2].setArg(1,sizeof(int),&N);
    r.kernels[2].setArg(2, sizeof(unsigned int),&width);
    r.kernels[2].setArg(3, sizeof(unsigned int),&one);

    r.kernels[3].setArg(0,sizeof(cl_mem), &x);
    r.kernels[3].setArg(1,sizeof(cl_mem), &y);
    r.kernels[3].setArg(2,sizeof(unsigned int), &N);
    r.kernels[3].setArg(3,sizeof(unsigned int), &width);

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


void FBLASEnvironment::sswap(std::string routine_name, unsigned int N, cl::Buffer x,
                             int incx, cl::Buffer y, int incy, std::vector<cl::Event> *events_wait_list, cl::Event *event)
{
    FBLASEnvironment::swap<float>(routine_name,N,x,incx,y,incy,events_wait_list,event);
}

void FBLASEnvironment::dswap(std::string routine_name, unsigned int N, cl::Buffer x,
                             int incx, cl::Buffer y, int incy, std::vector<cl::Event> *events_wait_list, cl::Event *event)
{
    FBLASEnvironment::swap<double>(routine_name,N,x,incx,y,incy,events_wait_list,event);
}


/*-------------
 * ROT
 * ---------------*/


template <typename T>
void FBLASEnvironment::rot(std::string routine_name, unsigned int N, cl::Buffer x, int incx, cl::Buffer y, int incy, T c, T s,
                           std::vector<cl::Event> * events_wait_list, cl::Event * event )
{
    //Set kernel args
    if(routines_.find(routine_name)==routines_.end())
    {
        std::cerr << "Routine "<<routine_name<<" not present" <<std::endl;
        return;
    }
    Routine r=routines_[routine_name];

    CHECK_INCX(incx,r);
    CHECK_INCY(incy,r);

    unsigned int width=r.width;
    unsigned int one=1;
    //set args: readers inject data with padding size equal to W

    r.kernels[0].setArg(0, sizeof(int),&N);
    r.kernels[0].setArg(1, sizeof(T),&c);
    r.kernels[0].setArg(2, sizeof(T),&s);
    r.kernels[1].setArg(0,sizeof(cl_mem), &x);
    r.kernels[1].setArg(1,sizeof(int),&N);
    r.kernels[1].setArg(2, sizeof(unsigned int),&width);
    r.kernels[1].setArg(3, sizeof(unsigned int),&one);

    r.kernels[2].setArg(0,sizeof(cl_mem), &y);
    r.kernels[2].setArg(1,sizeof(int),&N);
    r.kernels[2].setArg(2, sizeof(unsigned int),&width);
    r.kernels[2].setArg(3, sizeof(unsigned int),&one);

    r.kernels[3].setArg(0,sizeof(cl_mem), &x);
    r.kernels[3].setArg(1,sizeof(cl_mem), &y);
    r.kernels[3].setArg(2,sizeof(unsigned int), &N);
    r.kernels[3].setArg(3,sizeof(unsigned int), &width);

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

void FBLASEnvironment::srot(std::string routine_name, unsigned int N, cl::Buffer x, int incx, cl::Buffer y, int incy, float c, float s,
                            std::vector<cl::Event> * events_wait_list, cl::Event * event )
{
    FBLASEnvironment::rot<float>(routine_name,N,x,incx,y,incy,c,s,events_wait_list,event);
}

void FBLASEnvironment::drot(std::string routine_name, unsigned int N, cl::Buffer x, int incx, cl::Buffer y, int incy, double c, double s,
                            std::vector<cl::Event> * events_wait_list, cl::Event * event )
{
    FBLASEnvironment::rot<double>(routine_name,N,x,incx,y,incy,c,s,events_wait_list,event);
}


/*-----------------
 * ROTG
 *-----------------*/

template <typename T>
void FBLASEnvironment::rotg(std::string routine_name, T sa, T sb , T &c, T &s)
{
    //This is implemented directly into the API
    //(the check is useless, but it has been keeped for consistency with the rest)
    if(routines_.find(routine_name)==routines_.end())
    {
        std::cerr << "Routine "<<routine_name<<" not present" <<std::endl;
        return;
    }

    T roe = (fabs(sa) > fabs(sb)) ? sa : sb;
    T scale = (T)(fabs(sa)) + (T)(fabs(sb));
    T r, z;
    if(scale != 0)
    {
        T aos = sa/scale;
        T bos = sb/scale;
        r = scale * sqrt(aos * aos + bos * bos);
        r = (roe > 0 ) ? r : -r ;
        c = sa / r;
        s = sb / r;
        z = 1.0f;
        if (fabs(sa) > fabs(sb))
            z = s;
        if (fabs(sb) >= fabs(sa) && c != 0.0)
            z = (T)(1.0) / c;
    }
    else {
        c = ((T)1.0);
        s = ((T)0.0);
    }
}

void FBLASEnvironment::srotg(std::string routine_name, float sa, float sb, float &c, float &s)
{
    FBLASEnvironment::rotg<float>(routine_name,sa,sb,c,s);
}

void FBLASEnvironment::drotg(std::string routine_name, double sa, double sb, double &c, double &s)
{
    FBLASEnvironment::rotg<double>(routine_name,sa,sb,c,s);
}


/*---------------
 * ROTM
 *---------------*/

template <typename T>
void FBLASEnvironment::rotm(std::string routine_name, unsigned int N, cl::Buffer x, int incx, cl::Buffer y, int incy, T param[5],
std::vector<cl::Event> * events_wait_list, cl::Event * event)
{
    //Set kernel args
    if(routines_.find(routine_name)==routines_.end())
    {
        std::cerr << "Routine "<<routine_name<<" not present" <<std::endl;
        return;
    }
    Routine r=routines_[routine_name];
    CHECK_INCX(incx,r);
    CHECK_INCY(incy,r);

    unsigned int width=r.width;
    unsigned int one=1;
    //set args: readers inject data with padding size equal to W

    r.kernels[0].setArg(0, sizeof(int),&N);
    r.kernels[0].setArg(1, sizeof(T),&param[0]);
    r.kernels[0].setArg(2, sizeof(T),&param[1]);
    r.kernels[0].setArg(3, sizeof(T),&param[2]);
    r.kernels[0].setArg(4, sizeof(T),&param[3]);
    r.kernels[0].setArg(5, sizeof(T),&param[4]);

    r.kernels[1].setArg(0,sizeof(cl_mem), &x);
    r.kernels[1].setArg(1,sizeof(int),&N);
    r.kernels[1].setArg(2, sizeof(unsigned int),&width);
    r.kernels[1].setArg(3, sizeof(unsigned int),&one);

    r.kernels[2].setArg(0,sizeof(cl_mem), &y);
    r.kernels[2].setArg(1,sizeof(int),&N);
    r.kernels[2].setArg(2, sizeof(unsigned int),&width);
    r.kernels[2].setArg(3, sizeof(unsigned int),&one);

    r.kernels[3].setArg(0,sizeof(cl_mem), &x);
    r.kernels[3].setArg(1,sizeof(cl_mem), &y);
    r.kernels[3].setArg(2,sizeof(unsigned int), &N);
    r.kernels[3].setArg(3,sizeof(unsigned int), &width);

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


void FBLASEnvironment::srotm(std::string routine_name, unsigned int N, cl::Buffer x, int incx,
                             cl::Buffer y, int incy, float param[], std::vector<cl::Event> *events_wait_list, cl::Event *event)
{
    FBLASEnvironment::rotm<float>(routine_name,N,x,incx,y,incy,param,events_wait_list,event);
}


void FBLASEnvironment::drotm(std::string routine_name, unsigned int N, cl::Buffer x, int incx,
                             cl::Buffer y, int incy, double param[], std::vector<cl::Event> *events_wait_list, cl::Event *event)
{
    FBLASEnvironment::rotm<double>(routine_name,N,x,incx,y,incy,param,events_wait_list,event);
}


/*------------------
 * ROTMG
 * ----------------*/
template <typename T>
void FBLASEnvironment::rotmg(std::string routine_name, T &d1, T &d2, T &x1, T y1, T param[5])
{
    //This is implemented directly into the API
    //(the check is useless, but it has been keeped for consistency with the rest)
    if(routines_.find(routine_name)==routines_.end())
    {
        std::cerr << "Routine "<<routine_name<<" not present" <<std::endl;
        return;
    }
    T h1,h2,h3,h4,u;
    T x=x1,y=y1;
    const T g = 4096;
    const T g2 = g*g;
    if (d1 < 0.0) {
        param[0] = -1;
        param[1] = 0;
        param[2] = 0;
        param[3] = 0;
        param[4] = 0;
        d1 = 0;
        d2 = 0;
        x1 = 0;
        return;
    }

    if (d2 * y1 == 0.0) {
        param[0] = -2;
        return;
    }

    T c = (T)fabs(d1 * x * x);
    T s = (T)fabs(d2 * y* y);

    if (c > s) {
        param[0] = (T)0.0;

        h1 = 1;
        h2 = (d2 * y) / (d1 * x);
        h3 = -y / x;
        h4 = 1;

        u = 1 - h3 * h2;

        if (u <= 0.0) {             /* the case u <= 0 is rejected */
            param[0] = -1;
            param[1] = 0;
            param[2] = 0;
            param[3] = 0;
            param[4] = 0;
            d1 = 0;
            d2 = 0;
            x1 = 0;
            return;
        }

        d1 /= u;
        d2 /= u;
        x *= u;
    } else {
        /* case of equation A7 */

        if (d2 * y * y < 0.0) {
            param[0] = -1;
            param[1] = 0;
            param[2] = 0;
            param[3] = 0;
            param[4] = 0;
            d1 = 0;
            d2 = 0;
            x1 = 0;
            return;
        }

        param[0] = 1;

        h1 = (d1 * x) / (d2 * y);
        h2 = 1;
        h3 = -1;
        h4 = x / y;
        u = 1 + h1 * h4;
        d1 /= u;
        d2 /= u;
        T tmp = d2;
        d2 = d1;
        d1 = tmp;

        x = y * u;
    }

    /* rescale d1 to range [1/g2,g2] */

    while (d1 <= 1.0 / g2 && d1 != 0.0) {
        param[0] = -1;
        d1 *= g2;
        x /= g;
        h2 /= g;
        h2 /= g;
    }

    while (d1 >= g2) {
        param[0] = -1;
        d1 /= g2;
        x *= g;
        h1 *= g;
        h2 *= g;
    }

    /* rescale D2 to range [1/G2,G2] */

    while (fabs(d2) <= 1.0 / g2 && d2 != 0.0) {
        param[0] = -1;
        d2 *= g2;
        h3 /= g;
        h4 /= g;
    }

    while (fabs(d2) >= g2) {
        param[0] = -1;
        d2 /= g2;
        h3 *= g;
        h4 *= g;
    }
    x1=x;

    if (param[0] == -1.0) {
        param[1] = h1;
        param[2] = h3;
        param[3] = h2;
        param[4] = h4;
    } else if (param[0] == 0.0) {
        param[2] = h3;
        param[3] = h2;
    } else if (param[0] == 1.0) {
        param[1] = h1;
        param[4] = h4;
    }
}

void FBLASEnvironment::srotmg(std::string routine_name, float &sd1, float &sd2, float &sx1, float sy1, float param[])
{
    FBLASEnvironment::rotmg<float>(routine_name,sd1,sd2,sx1,sy1,param);
}

void FBLASEnvironment::drotmg(std::string routine_name, double &dd1, double &dd2, double &dx1, double dy1, double param[])
{
    FBLASEnvironment::rotmg<double>(routine_name,dd1,dd2,dx1,dy1,param);
}

/*-------------
 * AXPY
 * ---------------*/


template <typename T>
void FBLASEnvironment::axpy(std::string routine_name,  unsigned int N, T alpha, cl::Buffer x, int incx,  cl::Buffer y, int incy,
                            std::vector<cl::Event> * events_wait_list, cl::Event * event)
{
    //Set kernel args
    if(routines_.find(routine_name)==routines_.end())
    {
        std::cerr << "Routine "<<routine_name<<" not present" <<std::endl;
        return;
    }
    Routine r=routines_[routine_name];

    if(r.incx!=incx || r.incy!=incy)
    {
        std::cout<< "Error in callind axpy: incx/incy different with respect to the compiled version" <<std::endl;
        return;
    }

    unsigned int width=r.width;
    unsigned int one=1;
    //set args: readers inject data with padding size equal to W

    r.kernels[0].setArg(0, sizeof(T),&alpha);
    r.kernels[0].setArg(1, sizeof(int),&N);
    r.kernels[1].setArg(0,sizeof(cl_mem), &x);
    r.kernels[1].setArg(1,sizeof(int),&N);
    r.kernels[1].setArg(2, sizeof(unsigned int),&width);
    r.kernels[1].setArg(3, sizeof(unsigned int),&one);

    r.kernels[2].setArg(0,sizeof(cl_mem), &y);
    r.kernels[2].setArg(1,sizeof(int),&N);
    r.kernels[2].setArg(2, sizeof(unsigned int),&width);
    r.kernels[2].setArg(3, sizeof(unsigned int),&one);

    r.kernels[3].setArg(0,sizeof(cl_mem), &y);
    r.kernels[3].setArg(1,sizeof(unsigned int), &N);
    r.kernels[3].setArg(2,sizeof(unsigned int), &width);

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

void FBLASEnvironment::saxpy(std::string routine_name,  unsigned int N, float alpha, cl::Buffer x, int incx,  cl::Buffer y, int incy,
                             std::vector<cl::Event> * events_wait_list, cl::Event * event)
{
    FBLASEnvironment::axpy<float>(routine_name,N,alpha,x,incx,y,incy,events_wait_list,event);
}


void FBLASEnvironment::daxpy(std::string routine_name,  unsigned int N, double alpha, cl::Buffer x, int incx,  cl::Buffer y, int incy,
                             std::vector<cl::Event> * events_wait_list, cl::Event * event)
{
    FBLASEnvironment::axpy<double>(routine_name,N,alpha,x,incx,y,incy,events_wait_list,event);
}

/*-------------
 *  DOT
 *-------------*/

template <typename T>
void FBLASEnvironment::dot(std::string routine_name, unsigned int N, cl::Buffer x, int incx, cl::Buffer y, int incy, cl::Buffer res, std::vector<cl::Event> * events_wait_list, cl::Event * event)
{
    //Set kernel args
    if(routines_.find(routine_name)==routines_.end())
    {
        std::cerr << "Routine "<<routine_name<<" not present" <<std::endl;
        return;
    }
    Routine r=routines_[routine_name];

    if(r.incx!=incx || r.incy!=incy)
    {
        std::cout<< "Error in callind sdot: incx/incy different with respect to the compiled version" <<std::endl;
        return;
    }

    unsigned int width=r.width;
    unsigned int one=1;
    //set args: readers inject data with padding size equal to W

    r.kernels[0].setArg(0, sizeof(int),&N);
    r.kernels[1].setArg(0,sizeof(cl_mem), &x);
    r.kernels[1].setArg(1,sizeof(int),&N);
    r.kernels[1].setArg(2, sizeof(unsigned int),&width);
    r.kernels[1].setArg(3, sizeof(unsigned int),&one);

    r.kernels[2].setArg(0,sizeof(cl_mem), &y);
    r.kernels[2].setArg(1,sizeof(int),&N);
    r.kernels[2].setArg(2, sizeof(unsigned int),&width);
    r.kernels[2].setArg(3, sizeof(unsigned int),&one);
    r.kernels[3].setArg(0,sizeof(cl_mem), &res);

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


void FBLASEnvironment::sdot(std::string routine_name, unsigned int N, cl::Buffer x, int incx, cl::Buffer y, int incy, cl::Buffer res, std::vector<cl::Event> * events_wait_list, cl::Event * event)
{
    FBLASEnvironment::dot<float>(routine_name,N,x,incx,y,incy,res,events_wait_list,event);
}

void FBLASEnvironment::ddot(std::string routine_name, unsigned int N, cl::Buffer x, int incx, cl::Buffer y, int incy, cl::Buffer res, std::vector<cl::Event> * events_wait_list, cl::Event * event)
{
    FBLASEnvironment::dot<double>(routine_name,N,x,incx,y,incy,res,events_wait_list,event);
}


/*-------------------
 *  SCAL
 *-------------------*/

template <typename T>
void FBLASEnvironment::scal(std::string routine_name, unsigned int N, T alpha, cl::Buffer x,  int incx, std::vector<cl::Event> * events_wait_list, cl::Event * event)
{
    //Set kernel arguments: main kernel, vector reader and vector writer
    Routine r=routines_[routine_name];
    int one = 1;
    r.kernels[0].setArg(0, sizeof(unsigned int),&N);
    r.kernels[0].setArg(1, sizeof(T),&alpha);
    r.kernels[1].setArg(0,sizeof(cl_mem), &x);
    r.kernels[1].setArg(1, sizeof(unsigned int),&N);
    r.kernels[1].setArg(2, sizeof(unsigned int),&r.width);
    r.kernels[1].setArg(3, sizeof(unsigned int),&one);
    r.kernels[2].setArg(0,sizeof(cl_mem), &x);
    r.kernels[2].setArg(1,sizeof(unsigned int),&N);
    r.kernels[2].setArg(2, sizeof(unsigned int),&r.width);


    //launch kernels
    for(int i=0;i<r.kernels.size()-1;i++)
        r.queues[i].enqueueTask(r.kernels[i],events_wait_list);

    //launch the last one
    r.queues[r.kernels.size()-1].enqueueTask(r.kernels[r.kernels.size()-1],events_wait_list,event);

    if(!event)
        for(int i=0;i<r.kernels.size();i++)
            r.queues[i].finish();
}


void FBLASEnvironment::sscal(std::string routine_name, unsigned int N, float alpha, cl::Buffer x, int incx, std::vector<cl::Event> *events_wait_list, cl::Event *event)
{
    FBLASEnvironment::scal<float>(routine_name,N,alpha,x,incx,events_wait_list,event);
}

void FBLASEnvironment::dscal(std::string routine_name, unsigned int N, double alpha, cl::Buffer x, int incx, std::vector<cl::Event> *events_wait_list, cl::Event *event)
{
    FBLASEnvironment::scal<double>(routine_name,N,alpha,x,incx,events_wait_list,event);
}


/*-------------------
 *  COPY
 *-------------------*/

template <typename T>
void FBLASEnvironment::copy(std::string routine_name,  unsigned int N, cl::Buffer x, int incx,  cl::Buffer y, int incy, std::vector<cl::Event> * events_wait_list, cl::Event * event)
{
    //Set kernel arguments: main kernel, vector reader and vector writer
    Routine r=routines_[routine_name];
    int one = 1;
    r.kernels[0].setArg(0,sizeof(cl_mem), &x);
    r.kernels[0].setArg(1, sizeof(unsigned int),&N);
    r.kernels[0].setArg(2, sizeof(unsigned int),&r.width);
    r.kernels[0].setArg(3, sizeof(unsigned int),&one);
    r.kernels[1].setArg(0,sizeof(cl_mem), &y);
    r.kernels[1].setArg(1,sizeof(unsigned int),&N);
    r.kernels[1].setArg(2, sizeof(unsigned int),&r.width);


    //launch kernels
    for(int i=0;i<r.kernels.size()-1;i++)
        r.queues[i].enqueueTask(r.kernels[i],events_wait_list);

    //launch the last one
    r.queues[r.kernels.size()-1].enqueueTask(r.kernels[r.kernels.size()-1],events_wait_list,event);

    if(!event)
        for(int i=0;i<r.kernels.size();i++)
            r.queues[i].finish();
}


void FBLASEnvironment::scopy(std::string routine_name, unsigned int N, cl::Buffer x, int incx, cl::Buffer y, int incy, std::vector<cl::Event> *events_wait_list, cl::Event *event)
{
    FBLASEnvironment::copy<float>(routine_name,N,x,incx,y,incy,events_wait_list,event);
}

void FBLASEnvironment::dcopy(std::string routine_name, unsigned int N, cl::Buffer x, int incx, cl::Buffer y, int incy, std::vector<cl::Event> *events_wait_list, cl::Event *event)
{
    FBLASEnvironment::copy<double>(routine_name,N,x,incx,y,incy,events_wait_list,event);
}


/*--------------
 *  ASUM
 *---------------*/
template <typename T>
void FBLASEnvironment::asum(std::string routine_name, unsigned int N, cl::Buffer x, int incx, cl::Buffer res, std::vector<cl::Event> * events_wait_list , cl::Event * event)
{
    //template not really useful here...

    auto it= routines_.find(routine_name);
    if(it == routines_.end())
    {
        std::cerr << "There not exist a routine with this given name: "<<routine_name  << std::endl;
        return;
    }
    Routine r=it->second;

    //Set kernel arguments: main kernel, vector reader and vector writer (scalar)
    int one=1;
    r.kernels[0].setArg(0, sizeof(unsigned int),&N);
    r.kernels[1].setArg(0,sizeof(cl_mem), &x);
    r.kernels[1].setArg(1, sizeof(unsigned int),&N);
    r.kernels[1].setArg(2, sizeof(unsigned int),&r.width);
    r.kernels[1].setArg(3, sizeof(unsigned int),&one);
    r.kernels[2].setArg(0,sizeof(cl_mem),&res);

    //launch kernels
    for(int i=0;i<r.kernels.size()-1;i++)
        r.queues[i].enqueueTask(r.kernels[i],events_wait_list);

    //launch the last one
    r.queues[r.kernels.size()-1].enqueueTask(r.kernels[r.kernels.size()-1],events_wait_list,event);

    if(!event)
        for(int i=0;i<r.kernels.size();i++)
            r.queues[i].finish();
}

void FBLASEnvironment::sasum(std::string routine_name, unsigned int N, cl::Buffer x, int incx, cl::Buffer res, std::vector<cl::Event> *events_wait_list, cl::Event *event)
{
    FBLASEnvironment::asum<float>(routine_name,N,x,incx,res,events_wait_list,event);
}

void FBLASEnvironment::dasum(std::string routine_name, unsigned int N, cl::Buffer x, int incx, cl::Buffer res, std::vector<cl::Event> *events_wait_list, cl::Event *event)
{
    FBLASEnvironment::asum<double>(routine_name,N,x,incx,res,events_wait_list,event);
}


/*-----------
 * IAMAX
 *-----------*/

template <typename T>
void FBLASEnvironment::iamax(std::string routine_name, unsigned int N, cl::Buffer x, int incx, cl::Buffer res, std::vector<cl::Event> * events_wait_list , cl::Event * event)
{
    //template not really useful here...

    auto it= routines_.find(routine_name);
    if(it == routines_.end())
    {
        std::cerr << "There not exist a routine with this given name: "<<routine_name  << std::endl;
        return;
    }
    Routine r=it->second;
    CHECK_INCX(incx,r);
    //Set kernel arguments: main kernel, vector reader and vector writer (scalar)
    int one=1;
    r.kernels[0].setArg(0, sizeof(unsigned int),&N);
    r.kernels[1].setArg(0,sizeof(cl_mem), &x);
    r.kernels[1].setArg(1, sizeof(unsigned int),&N);
    r.kernels[1].setArg(2, sizeof(unsigned int),&r.width);
    r.kernels[1].setArg(3, sizeof(unsigned int),&one);
    r.kernels[2].setArg(0,sizeof(cl_mem),&res);

    //launch kernels
    for(int i=0;i<r.kernels.size()-1;i++)
        r.queues[i].enqueueTask(r.kernels[i],events_wait_list);

    //launch the last one
    r.queues[r.kernels.size()-1].enqueueTask(r.kernels[r.kernels.size()-1],events_wait_list,event);

    if(!event)
        for(int i=0;i<r.kernels.size();i++)
            r.queues[i].finish();
}

void FBLASEnvironment::isamax(std::string routine_name, unsigned int N, cl::Buffer x, int incx, cl::Buffer res, std::vector<cl::Event> *events_wait_list, cl::Event *event)
{
    FBLASEnvironment::iamax<float>(routine_name,N,x,incx,res,events_wait_list,event);
}

void FBLASEnvironment::idamax(std::string routine_name, unsigned int N, cl::Buffer x, int incx, cl::Buffer res, std::vector<cl::Event> *events_wait_list, cl::Event *event)
{
    FBLASEnvironment::iamax<double>(routine_name,N,x,incx,res,events_wait_list,event);
}

/*---------
 * NRM2
 * --------*/

template <typename T>
void FBLASEnvironment::nrm2(std::string routine_name, unsigned int N, cl::Buffer x, int incx, cl::Buffer res, std::vector<cl::Event> * events_wait_list , cl::Event * event)
{
    //template not really useful here...

    auto it= routines_.find(routine_name);
    if(it == routines_.end())
    {
        std::cerr << "There not exist a routine with this given name: "<<routine_name  << std::endl;
        return;
    }
    Routine r=it->second;

    //Set kernel arguments: main kernel, vector reader and vector writer (scalar)
    int one=1;
    r.kernels[0].setArg(0, sizeof(unsigned int),&N);
    r.kernels[1].setArg(0,sizeof(cl_mem), &x);
    r.kernels[1].setArg(1, sizeof(unsigned int),&N);
    r.kernels[1].setArg(2, sizeof(unsigned int),&r.width);
    r.kernels[1].setArg(3, sizeof(unsigned int),&one);
    r.kernels[2].setArg(0,sizeof(cl_mem),&res);

    //launch kernels
    for(int i=0;i<r.kernels.size()-1;i++)
        r.queues[i].enqueueTask(r.kernels[i],events_wait_list);

    //launch the last one
    r.queues[r.kernels.size()-1].enqueueTask(r.kernels[r.kernels.size()-1],events_wait_list,event);

    if(!event)
        for(int i=0;i<r.kernels.size();i++)
            r.queues[i].finish();
}

void FBLASEnvironment::snrm2(std::string routine_name, unsigned int N, cl::Buffer x, int incx, cl::Buffer res, std::vector<cl::Event> *events_wait_list, cl::Event *event)
{
    FBLASEnvironment::asum<float>(routine_name,N,x,incx,res,events_wait_list,event);
}

void FBLASEnvironment::dnrm2(std::string routine_name, unsigned int N, cl::Buffer x, int incx, cl::Buffer res, std::vector<cl::Event> *events_wait_list, cl::Event *event)
{
    FBLASEnvironment::asum<double>(routine_name,N,x,incx,res,events_wait_list,event);
}





#endif


