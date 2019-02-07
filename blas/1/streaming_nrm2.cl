/**

    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2019 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    NRM2 returns the euclidean norm of a vector via the function
    name, so that
            NRM2 := sqrt( x'*x )

    Data is received from the input stream CHANNEL_VECTOR_X.
    Data elements must be streamed with a padding equal to W
    (padding data must be set to zero).

    Result is streamed in an output channel at the end of the computation
    in a channel CHANNEL_OUT.

    If DOUBLE_PRECISION is defined, the routine computes DNRM2
    otherwise SRNM2

*/

#pragma OPENCL EXTENSION cl_intel_channels : enable
//FBLAS_PARAMETERS_START


//#define DOUBLE_PRECISION  //define this for DNRM2
#define W 32		    //width


//namings
#define KERNEL_NAME streaming_nrm2

//channel definitions
#define CHANNEL_VECTOR_X channel_vector_x
#define CHANNEL_OUT channel_sink

//architecture
#define __STRATIX_10__

//FBLAS_PARAMETERS_END

#include <commons.h>

channel TYPE_T CHANNEL_VECTOR_X __attribute__((depth(W)));
channel TYPE_T CHANNEL_OUT __attribute__((depth(1)));


__kernel void KERNEL_NAME(int N)
{

    const int outer_loop_limit=1+(int)((N-1)/W); //ceiling
    TYPE_T acc_o=0;
    TYPE_T x[W];
    #ifdef DOUBLE_PRECISION
        TYPE_T shift_reg[SHIFT_REG+1]; //shift register

        for(int i=0;i<SHIFT_REG+1;i++)
           shift_reg[i]=0;
    #endif

    for(int i=0; i<outer_loop_limit; i++)
    {
        //By unrolling this two loops, we perform W mult. per cycle
        TYPE_T acc_i=0;
        #pragma unroll
        for(int j=0;j<W;j++)
        {
            x[j]=read_channel_intel(CHANNEL_VECTOR_X);
            acc_i+=x[j]*x[j];
        }
        #ifdef DOUBLE_PRECISION
            shift_reg[SHIFT_REG] = shift_reg[0]+acc_i;
            //Shift every element of shift register
            #pragma unroll
            for(int j = 0; j < SHIFT_REG; ++j)
                shift_reg[j] = shift_reg[j + 1];

        #else
            acc_o+=acc_i;
        #endif

    }
    #ifdef DOUBLE_PRECISION
        #pragma unroll
        for(int i=0;i<SHIFT_REG;i++)
            acc_o+=shift_reg[i];
    #endif

    TYPE_T res=sqrt(acc_o);
    write_channel_intel(CHANNEL_OUT,res);

}
