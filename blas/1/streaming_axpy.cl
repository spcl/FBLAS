/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2019 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    AXPY constant times a vector plus a vector.

    Streamed version: data is received from two input streams
    CHANNEL_VECTOR_X and CHANNEL_VECTOR_Y having the proper type.
    Data elements must be streamed with a padding equal to W
    (padding data must be set to zero).

    Result is streamed in an output channel at the end of the computation
    in a channel CHANNEL_OUT

*/


#pragma OPENCL EXTENSION cl_intel_channels : enable

//FBLAS_PARAMETERS_START

#define DOUBLE_PRECISION  //enable for daxpy
#define W 64		    //width: number of multiplications performed per clock cycle

//namings
#define KERNEL_NAME sreaming_saxpy

//channels names
#define CHANNEL_VECTOR_X channel_gen_x
#define CHANNEL_VECTOR_Y channel_gen_y
#define CHANNEL_VECTOR_OUT channel_sink

//architecture
#define __STRATIX_10__

//FBLAS_PARAMETERS_END

#include <commons.h>
channel TYPE_T CHANNEL_VECTOR_X __attribute__((depth(W)));
channel TYPE_T CHANNEL_VECTOR_Y __attribute__((depth(W)));
channel TYPE_T CHANNEL_VECTOR_OUT __attribute__((depth(W)));

__kernel void KERNEL_NAME(const TYPE_T alpha, int N)
{

    if(N==0) return;

    const int outer_loop_limit=1+(int)((N-1)/W); //ceiling
    TYPE_T res[W];

    for(int i=0; i<outer_loop_limit; i++)
    {
        //receive W elements from the input channels
        #pragma unroll
        for(int j=0;j<W;j++)
            res[j]=alpha*read_channel_intel(CHANNEL_VECTOR_X)+read_channel_intel(CHANNEL_VECTOR_Y);

        //sends the data to a writer
        #pragma unroll
        for(int j=0; j<W; j++)
            write_channel_intel(CHANNEL_VECTOR_OUT,res[j]);
    }

}
