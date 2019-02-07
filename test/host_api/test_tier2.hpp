/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2019 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    This file contains a set of utilities function and constant used for testing
*/
#ifndef DATA_GENERATORS_HPP
#define DATA_GENERATORS_HPP
#include <iostream>

#define MAX_NUMB 10.0       //max value of a numerical entry
#define OFFSET(N, incx) ((incx) > 0 ?  0 : ((N) - 1) * (-(incx)))

//Constants and data used in testing Level 1 (source Netlib, *blat1.f / *blat1.c)

static int ns[4] = { 0,1,2,4 };
static const int N_L1=7;
const int nincs=4;
static int incxs[4] = { 1,2,-2,-1 };
static int incys[4] = { 1,-2,1,-2 };
const int max_inc=2;                    //max value for a vector stride

const float dx1[7] = { .6f,.1f,-.5f,.8f,.9f,-.3f,-.4f };
const float dy1[7] = { .5f,-.9f,.3f,.7f,-.6f,.2f,.8f };
const float dt10y[112]	/* was [7][4][4] */ = { .5f,0.f,0.f,0.f,0.f,
                                                0.f,0.f,.6f,0.f,0.f,0.f,0.f,0.f,0.f,.6f,.1f,0.f,0.f,0.f,0.f,0.f,
                                                .6f,.1f,-.5f,.8f,0.f,0.f,0.f,.5f,0.f,0.f,0.f,0.f,0.f,0.f,.6f,0.f,
                                                0.f,0.f,0.f,0.f,0.f,-.5f,-.9f,.6f,0.f,0.f,0.f,0.f,-.4f,-.9f,.9f,
                                                .7f,-.5f,.2f,.6f,.5f,0.f,0.f,0.f,0.f,0.f,0.f,.6f,0.f,0.f,0.f,0.f,
                                                0.f,0.f,-.5f,.6f,0.f,0.f,0.f,0.f,0.f,-.4f,.9f,-.5f,.6f,0.f,0.f,
                                                0.f,.5f,0.f,0.f,0.f,0.f,0.f,0.f,.6f,0.f,0.f,0.f,0.f,0.f,0.f,.6f,
                                                -.9f,.1f,0.f,0.f,0.f,0.f,.6f,-.9f,.1f,.7f,-.5f,.2f,.8f };
const float dt10x[112]	/* was [7][4][4] */ = { .6f,0.f,0.f,0.f,0.f,
                                                0.f,0.f,.5f,0.f,0.f,0.f,0.f,0.f,0.f,.5f,-.9f,0.f,0.f,0.f,0.f,0.f,
                                                .5f,-.9f,.3f,.7f,0.f,0.f,0.f,.6f,0.f,0.f,0.f,0.f,0.f,0.f,.5f,0.f,
                                                0.f,0.f,0.f,0.f,0.f,.3f,.1f,.5f,0.f,0.f,0.f,0.f,.8f,.1f,-.6f,.8f,
                                                .3f,-.3f,.5f,.6f,0.f,0.f,0.f,0.f,0.f,0.f,.5f,0.f,0.f,0.f,0.f,0.f,
                                                0.f,-.9f,.1f,.5f,0.f,0.f,0.f,0.f,.7f,.1f,.3f,.8f,-.9f,-.3f,.5f,
                                                .6f,0.f,0.f,0.f,0.f,0.f,0.f,.5f,0.f,0.f,0.f,0.f,0.f,0.f,.5f,.3f,
                                                0.f,0.f,0.f,0.f,0.f,.5f,.3f,-.6f,.8f,0.f,0.f,0.f };

const float dt9x[112]	/* was [7][4][4] */ = { .6f,0.f,0.f,0.f,0.f,
                                                0.f,0.f,.78f,0.f,0.f,0.f,0.f,0.f,0.f,.78f,-.46f,0.f,0.f,0.f,0.f,
                                                0.f,.78f,-.46f,-.22f,1.06f,0.f,0.f,0.f,.6f,0.f,0.f,0.f,0.f,0.f,
                                                0.f,.78f,0.f,0.f,0.f,0.f,0.f,0.f,.66f,.1f,-.1f,0.f,0.f,0.f,0.f,
                                                .96f,.1f,-.76f,.8f,.9f,-.3f,-.02f,.6f,0.f,0.f,0.f,0.f,0.f,0.f,
                                                .78f,0.f,0.f,0.f,0.f,0.f,0.f,-.06f,.1f,-.1f,0.f,0.f,0.f,0.f,.9f,
                                                .1f,-.22f,.8f,.18f,-.3f,-.02f,.6f,0.f,0.f,0.f,0.f,0.f,0.f,.78f,
                                                0.f,0.f,0.f,0.f,0.f,0.f,.78f,.26f,0.f,0.f,0.f,0.f,0.f,.78f,.26f,
                                                -.76f,1.12f,0.f,0.f,0.f };
const float dt9y[112]	/* was [7][4][4] */ = { .5f,0.f,0.f,0.f,0.f,
                                                0.f,0.f,.04f,0.f,0.f,0.f,0.f,0.f,0.f,.04f,-.78f,0.f,0.f,0.f,0.f,
                                                0.f,.04f,-.78f,.54f,.08f,0.f,0.f,0.f,.5f,0.f,0.f,0.f,0.f,0.f,0.f,
                                                .04f,0.f,0.f,0.f,0.f,0.f,0.f,.7f,-.9f,-.12f,0.f,0.f,0.f,0.f,.64f,
                                                -.9f,-.3f,.7f,-.18f,.2f,.28f,.5f,0.f,0.f,0.f,0.f,0.f,0.f,.04f,0.f,
                                                0.f,0.f,0.f,0.f,0.f,.7f,-1.08f,0.f,0.f,0.f,0.f,0.f,.64f,-1.26f,
                                                .54f,.2f,0.f,0.f,0.f,.5f,0.f,0.f,0.f,0.f,0.f,0.f,.04f,0.f,0.f,0.f,
                                                0.f,0.f,0.f,.04f,-.9f,.18f,0.f,0.f,0.f,0.f,.04f,-.9f,.18f,.7f,
                                                -.18f,.2f,.16f };


const float da1[8] = { .3f,.4f,-.3f,-.4f,-.3f,0.f,0.f,1.f };
const float db1[8] = { .4f,.3f,.4f,.3f,-.4f,0.f,1.f,0.f };
const float dc1[8] = { .6f,.8f,-.6f,.8f,.6f,1.f,0.f,1.f };
const float ds1[8] = { .8f,.6f,.8f,-.6f,.8f,0.f,1.f,0.f };

const float dab[36]	/* was [4][9] */ = { .1f,.3f,1.2f,.2f,.7f,.2f,.6f,
                                             4.2f,0.f,0.f,0.f,0.f,4.f,-1.f,2.f,4.f,6e-10f,.02f,1e5f,10.f,4e10f,
                                             .02f,1e-5f,10.f,2e-10f,.04f,1e5f,10.f,2e10f,.04f,1e-5f,10.f,4.f,
                                             -2.f,8.f,4.f };

const float dtrue[81]	/* was [9][9] */ = { 0.f,0.f,1.3f,.2f,0.f,0.f,
           0.f,.5f,0.f,0.f,0.f,4.5f,4.2f,1.f,.5f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,
           -2.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,4.f,-1.f,0.f,0.f,0.f,0.f,0.f,
           .015f,0.f,10.f,-1.f,0.f,-1e-4f,0.f,1.f,0.f,0.f,.06144f,10.f,-1.f,
           4096.f,-1e6f,0.f,1.f,0.f,0.f,15.f,10.f,-1.f,5e-5f,0.f,1.f,0.f,0.f,
           0.f,15.f,10.f,-1.f,5e5f,-4096.f,1.f,.004096f,0.f,0.f,7.f,4.f,0.f,
           0.f,-.5f,-.25f,0.f };

const float dv[80]	/* was [8][5][2] */ = { .1f,2.f,2.f,2.f,2.f,2.f,2.f,
           2.f,.3f,3.f,3.f,3.f,3.f,3.f,3.f,3.f,.3f,-.4f,4.f,4.f,4.f,4.f,4.f,
           4.f,.2f,-.6f,.3f,5.f,5.f,5.f,5.f,5.f,.1f,-.3f,.5f,-.1f,6.f,6.f,
           6.f,6.f,.1f,8.f,8.f,8.f,8.f,8.f,8.f,8.f,.3f,9.f,9.f,9.f,9.f,9.f,
           9.f,9.f,.3f,2.f,-.4f,2.f,2.f,2.f,2.f,2.f,.2f,3.f,-.6f,5.f,.3f,2.f,
           2.f,2.f,.1f,4.f,-.3f,6.f,-.5f,7.f,-.1f,3.f };

const int itrue2[5] = { 0,1,2,2,3 };



const double double_dx1[7] = { .6,.1,-.5,.8,.9,-.3,-.4 };
const double double_dy1[7] = { .5,-.9,.3,.7,-.6,.2,.8 };

const double double_dt10x[112]	/* was [7][4][4] */ = { .6,0.,0.,0.,
                                                        0.,0.,0.,.5,0.,0.,0.,0.,0.,0.,.5,-.9,0.,0.,0.,0.,0.,.5,-.9,.3,.7,
                                                        0.,0.,0.,.6,0.,0.,0.,0.,0.,0.,.5,0.,0.,0.,0.,0.,0.,.3,.1,.5,0.,0.,
                                                        0.,0.,.8,.1,-.6,.8,.3,-.3,.5,.6,0.,0.,0.,0.,0.,0.,.5,0.,0.,0.,0.,
                                                        0.,0.,-.9,.1,.5,0.,0.,0.,0.,.7,.1,.3,.8,-.9,-.3,.5,.6,0.,0.,0.,0.,
                                                        0.,0.,.5,0.,0.,0.,0.,0.,0.,.5,.3,0.,0.,0.,0.,0.,.5,.3,-.6,.8,0.,
                                                        0.,0. };
const double double_dt10y[112]	/* was [7][4][4] */ = { .5,0.,0.,0.,
                                                        0.,0.,0.,.6,0.,0.,0.,0.,0.,0.,.6,.1,0.,0.,0.,0.,0.,.6,.1,-.5,.8,
                                                        0.,0.,0.,.5,0.,0.,0.,0.,0.,0.,.6,0.,0.,0.,0.,0.,0.,-.5,-.9,.6,0.,
                                                        0.,0.,0.,-.4,-.9,.9,.7,-.5,.2,.6,.5,0.,0.,0.,0.,0.,0.,.6,0.,0.,0.,
                                                        0.,0.,0.,-.5,.6,0.,0.,0.,0.,0.,-.4,.9,-.5,.6,0.,0.,0.,.5,0.,0.,0.,
                                                        0.,0.,0.,.6,0.,0.,0.,0.,0.,0.,.6,-.9,.1,0.,0.,0.,0.,.6,-.9,.1,.7,
                                                        -.5,.2,.8 };


const double double_dt9x[112]	/* was [7][4][4] */ = { .6,0.,0.,0.,0.,0.,0.,
                                                        .78,0.,0.,0.,0.,0.,0.,.78,-.46,0.,0.,0.,0.,0.,.78,-.46,-.22,1.06,
                                                        0.,0.,0.,.6,0.,0.,0.,0.,0.,0.,.78,0.,0.,0.,0.,0.,0.,.66,.1,-.1,0.,
                                                        0.,0.,0.,.96,.1,-.76,.8,.9,-.3,-.02,.6,0.,0.,0.,0.,0.,0.,.78,0.,
                                                        0.,0.,0.,0.,0.,-.06,.1,-.1,0.,0.,0.,0.,.9,.1,-.22,.8,.18,-.3,-.02,
                                                        .6,0.,0.,0.,0.,0.,0.,.78,0.,0.,0.,0.,0.,0.,.78,.26,0.,0.,0.,0.,0.,
                                                        .78,.26,-.76,1.12,0.,0.,0. };
const double double_dt9y[112]	/* was [7][4][4] */ = { .5,0.,0.,0.,0.,0.,0.,
                                                        .04,0.,0.,0.,0.,0.,0.,.04,-.78,0.,0.,0.,0.,0.,.04,-.78,.54,.08,0.,
                                                        0.,0.,.5,0.,0.,0.,0.,0.,0.,.04,0.,0.,0.,0.,0.,0.,.7,-.9,-.12,0.,
                                                        0.,0.,0.,.64,-.9,-.3,.7,-.18,.2,.28,.5,0.,0.,0.,0.,0.,0.,.04,0.,
                                                        0.,0.,0.,0.,0.,.7,-1.08,0.,0.,0.,0.,0.,.64,-1.26,.54,.2,0.,0.,0.,
                                                        .5,0.,0.,0.,0.,0.,0.,.04,0.,0.,0.,0.,0.,0.,.04,-.9,.18,0.,0.,0.,
                                                        0.,.04,-.9,.18,.7,-.18,.2,.16 };


const double double_da1[8] = { .3,.4,-.3,-.4,-.3,0.,0.,1. };
const double double_db1[8] = { .4,.3,.4,.3,-.4,0.,1.,0. };
const double double_dc1[8] = { .6,.8,-.6,.8,.6,1.,0.,1. };
const double double_ds1[8] = { .8,.6,.8,-.6,.8,0.,1.,0. };


// Constants used in testing (Level 2 and Level 3 routines)
const char icht[2] = {'N','T'};      //non-transposed, transposed
const char ichu[2] = {'L','U'};      //lower, upper
const char ichs[2] = {'L','R'};      //side, left and righ

//sizes
const int N=64;                 // max {N,M,K} size
const int M=N;
const int nd=3;
const int ndim[3]={4,16,64};    //sizes


//Constants used for evaluating correctness of result
const float flteps= 1e-4;
const float dbleps= 1e-6;


// Data generations utilities

template <typename T>
void generate_matrix(T *A,int N,int M)
{
    for(int i=0;i<N;i++)
    {
        for(int j=0;j<M;j++)
            A[i*M+j] = 1;// static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/MAX_NUMB));
    }
}

template <typename T>
void generate_vector(T *x,int N)
{
    for(int i=0;i<N;i++)
        x[i]= i+1;// static_cast <T> (rand()) / (static_cast <T> (RAND_MAX/MAX_NUMB));
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
