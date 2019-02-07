/**

    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2019 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    SWAP - "Interchanges" two stream

    Data is received from two input streams
    CHANNEL_VECTOR_X and CHANNEL_VECTOR_Y having the proper type.
    Data elements must be streamed with a padding equal to W
    (padding data must be set to zero).

    The content of the stream CHANNEL_VECTOR_X will be sent to CHANNEL_OUT_Y
    while the content of CHANNEL_VECTOR_Y will be sent to CHANNEL_OUT_X

*/


#pragma OPENCL EXTENSION cl_intel_channels : enable

//FBLAS_PARAMETERS_START

//#define DOUBLE_PRECISION  //enable for dswap
#define W 64                //width

//namings
#define KERNEL_NAME streaming_rot

//channel name definitions
#define CHANNEL_VECTOR_X channel_x
#define CHANNEL_VECTOR_Y channel_y
#define CHANNEL_VECTOR_OUT_X channel_out_x
#define CHANNEL_VECTOR_OUT_Y channel_out_y

//architecture
#define __STRATIX_10__

//FBLAS_PARAMETERS_END

#include <commons.h>

channel TYPE_T CHANNEL_VECTOR_X __attribute__((depth(W)));
channel TYPE_T CHANNEL_VECTOR_Y __attribute__((depth(W)));
channel TYPE_T CHANNEL_VECTOR_OUT_X __attribute__((depth(W)));
channel TYPE_T CHANNEL_VECTOR_OUT_Y __attribute__((depth(W)));

__kernel void KERNEL_NAME(const unsigned int N)
{
    if(N==0) return;

    const int outer_loop_limit=1+(int)((N-1)/W); //ceiling
    TYPE_T x[W];
    TYPE_T y[W];


    for(int i=0; i<outer_loop_limit; i++)
    {
        #pragma unroll
        for(int j=0;j<W;j++)
        {
            x[j]=read_channel_intel(CHANNEL_VECTOR_X);
            y[j]=read_channel_intel(CHANNEL_VECTOR_Y);
            write_channel_intel(CHANNEL_VECTOR_OUT_Y,x[j]);
            write_channel_intel(CHANNEL_VECTOR_OUT_X,y[j]);
        }
    }
}
