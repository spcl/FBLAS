/*

    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2019 ETH-Zurich. All rights reserved.
    See LICENSE for license information.


    ROT - applies a plan rotation

    Data is received from two input streams
    CHANNEL_VECTOR_X and CHANNEL_VECTOR_Y having the proper type.
    Data elements must be streamed with a padding equal to W
    (padding data must be set to zero).

    Result is streamed in an output channel at the end of the computation
    in two channels CHANNEL_OUT_X (for x) and CHANNEL_OUT_Y (for y)

*/


#pragma OPENCL EXTENSION cl_intel_channels : enable

//FBLAS_PARAMETERS_START

//#define DOUBLE_PRECISION      //enable for drot
#define W 64		    	//width

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

__kernel void KERNEL_NAME(const int N, const TYPE_T c, const TYPE_T s)
{

    if(N==0) return;
    const int outer_loop_limit=1+(int)((N-1)/W); //ceiling
    TYPE_T x[W];
    TYPE_T y[W];
    TYPE_T out_x[W];
    TYPE_T out_y[W];

    for(int i=0; i<outer_loop_limit; i++)
    {
        #pragma unroll
        for(int j=0;j<W;j++)
        {
            x[j]=read_channel_intel(CHANNEL_VECTOR_X);
            y[j]=read_channel_intel(CHANNEL_VECTOR_Y);

            out_x[j] = c * x[j] + s * y[j];
            out_y[j] = -s * x[j] + c * y[j];
            write_channel_intel(CHANNEL_VECTOR_OUT_X,out_x[j]);
            write_channel_intel(CHANNEL_VECTOR_OUT_Y,out_y[j]);
        }
    }
}
