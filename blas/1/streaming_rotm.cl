/**

    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2019 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    ROTM - applies the modifiet Givens rotation

    Data is received from two input streams
    CHANNEL_VECTOR_X and CHANNEL_VECTOR_Y having the proper type.
    Data elements must be streamed with a padding equal to W
    (padding data must be set to zero).

    Result is streamed in an output channel at the end of the computation
    in two channels CHANNEL_OUT_X (for x) and CHANNEL_OUT_Y (for y)

*/


#pragma OPENCL EXTENSION cl_intel_channels : enable

//FBLAS_PARAMETERS_START

//#define DOUBLE_PRECISION      //enable for drotm
#define W 64		    	//width

//namings
#define KERNEL_NAME streaming_rotm

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

__kernel void KERNEL_NAME(const int N, const TYPE_T flag, TYPE_T h11, TYPE_T h21, TYPE_T h12, TYPE_T h22)
{

    if(N==0) return;
    //Flag, h11, h21,h12 and h22 represent the "param" array of BLAS interface
    const int outer_loop_limit=1+(int)((N-1)/W); //ceiling
    TYPE_T x[W];
    TYPE_T y[W];
    TYPE_T out_x[W];
    TYPE_T out_y[W];

    //setup rotation

    if (flag == (TYPE_T)(0.0f)) {
        h11 = (TYPE_T)(1.0f);
        h22 = (TYPE_T)(1.0f);
    } else if (flag == (TYPE_T)(1.0f)) {
        h21 = (TYPE_T)(-1.0f);
        h12 = (TYPE_T)(1.0f);
    }
     else
        if (flag != (TYPE_T)(-1.0f))
            return;


    for(int i=0; i<outer_loop_limit; i++)
    {
        #pragma unroll
        for(int j=0;j<W;j++)
        {
            x[j]=read_channel_intel(CHANNEL_VECTOR_X);
            y[j]=read_channel_intel(CHANNEL_VECTOR_Y);

            out_x[j] = h11 * x[j] + h12 * y[j];
            out_y[j] = h21 * x[j] + h22 * y[j];
            write_channel_intel(CHANNEL_VECTOR_OUT_X,out_x[j]);
            write_channel_intel(CHANNEL_VECTOR_OUT_Y,out_y[j]);

        }
    }
}
