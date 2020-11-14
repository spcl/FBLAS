 /**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2020 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    DOT performs the dot product of two vectors.

    Streamed version: data is received from two input streams
    CHANNEL_VECTOR_X and CHANNEL_VECTOR_Y having the proper type.
    Data elements must be streamed with a padding equal to W
    (padding data must be set to zero).

    Result is streamed in an output channel at the end of the computation
    in a channel CHANNEL_OUT

*/

#pragma OPENCL EXTENSION cl_intel_channels : enable


#define W 64		    //width: number of multiplications performed per clock cycle

//namings
#define my_routine streaming_dot

//channels names
#define CHANNEL_VECTOR_X channel_gen_x
#define CHANNEL_VECTOR_Y channel_gen_y
#define CHANNEL_OUT channel_sink

//architecture
#define __STRATIX_10__

//FBLAS_PARAMETERS_END

#include <commons.h>

channel float CHANNEL_VECTOR_X __attribute__((depth(1)));
channel float CHANNEL_VECTOR_Y __attribute__((depth(1)));
channel float CHANNEL_OUT __attribute__((depth(1)));


/**
    Performs streaming dot product: data is received through
    CHANNEL_VECTOR_X and CHANNEL_VECTOR_Y. Result is sent
    to CHANNEL_OUT.
*/
__kernel void my_routine(int N)
{
    __constant uint WIDTH = 1
    float acc_o=0;
    if(N>0)
    {

        const int outer_loop_limit=1+(int)((N-1)/WIDTH); //ceiling
        float x[WIDTH],y[WIDTH];


        //Strip mine the computation loop to exploit unrolling
        for(int i=0; i<outer_loop_limit; i++)
        {

            float acc_i=0;
            #pragma unroll
            for(int j=0;j<WIDTH;j++)
            {
                x[j]=read_channel_intel(CHANNEL_VECTOR_X);
                y[j]=read_channel_intel(CHANNEL_VECTOR_Y);
                acc_i+=x[j]*y[j];

            }

                acc_o+=acc_i;

        }

    }
    else //no computation: result is zero
        acc_o=0.0f;
    //write to the sink
    write_channel_intel(CHANNEL_OUT,acc_o);
}