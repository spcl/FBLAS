 /**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2019 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    DOT performs the dot product of two vectors.

    If the DOUBLE_PRECISION macro is defined, it is computed over double numbers,
    otherwise float data is used.

    Streamed version: data is received from two input streams
    CHANNEL_VECTOR_X and CHANNEL_VECTOR_Y having the proper type.
    Data elements must be streamed with a padding equal to W
    (padding data must be set to zero).

    Result is streamed in an output channel at the end of the computation
    in a channel CHANNEL_OUT

*/

#pragma OPENCL EXTENSION cl_intel_channels : enable

//FBLAS_PARAMETERS_START

//double precision
//#define DOUBLE_PRECISION
#define W 64		    //width: number of multiplications performed per clock cycle

//namings
#define KERNEL_NAME streaming_dot

//channels names
#define CHANNEL_VECTOR_X channel_gen_x
#define CHANNEL_VECTOR_Y channel_gen_y
#define CHANNEL_OUT channel_sink

//architecture
#define __STRATIX_10__

//FBLAS_PARAMETERS_END

#include <commons.h>

channel TYPE_T CHANNEL_VECTOR_X __attribute__((depth(W)));
channel TYPE_T CHANNEL_VECTOR_Y __attribute__((depth(W)));
channel TYPE_T CHANNEL_OUT __attribute__((depth(1)));


/**
    Performs streaming dot product: data is received through
    CHANNEL_VECTOR_X and CHANNEL_VECTOR_Y. Result is sent
    to CHANNEL_OUT.
*/
__kernel void KERNEL_NAME(int N)
{
    TYPE_T acc_o=0;
    if(N>0)
    {

        const int outer_loop_limit=1+(int)((N-1)/W); //ceiling
        TYPE_T x[W],y[W];

        #ifdef DOUBLE_PRECISION
        TYPE_T shift_reg[SHIFT_REG+1]; //shift register

        for(int i=0;i<SHIFT_REG+1;i++)
           shift_reg[i]=0;
        #endif

        //Strip mine the computation loop to exploit unrolling
        for(int i=0; i<outer_loop_limit; i++)
        {

            TYPE_T acc_i=0;
            #pragma unroll
            for(int j=0;j<W;j++)
            {
                x[j]=read_channel_intel(CHANNEL_VECTOR_X);
                y[j]=read_channel_intel(CHANNEL_VECTOR_Y);
                acc_i+=x[j]*y[j];

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
            //reconstruct the result using the partial results in shift register
            #pragma unroll
            for(int i=0;i<SHIFT_REG;i++)
                acc_o+=shift_reg[i];
        #endif
    }
    else //no computation: result is zero
        acc_o=0.0f;
    //write to the sink
    write_channel_intel(CHANNEL_OUT,acc_o);
}
