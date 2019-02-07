/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2019 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    IAMAX finds the index of the first element having maximum absolute value

    Data is received from the input stream CHANNEL_VECTOR_X.
    Data elements must be streamed with a padding equal to W
    (padding data must be set to zero).

    Result is streamed in an output channel at the end of the computation
    in a channel CHANNEL_OUT

*/
#pragma OPENCL EXTENSION cl_intel_channels : enable

//FBLAS_PARAMETERS_START
//#define DOUBLE_PRECISION  //enable for IDAMAX
#define W 32		    //width

//namings
#define KERNEL_NAME streaming_iamax
#define CHANNEL_VECTOR_X channel_vector_x
#define CHANNEL_OUT channel_sink

//architecture
#define __STRATIX_10__

//FBLAS_PARAMETERS_END

#include <commons.h>
channel TYPE_T CHANNEL_VECTOR_X __attribute__((depth(W)));
channel int CHANNEL_OUT __attribute__((depth(1)));


__kernel void KERNEL_NAME(int N)
{

    const int outer_loop_limit=1+(int)((N-1)/W); //ceiling
    TYPE_T x[W];
    int g_max_index=0;
    TYPE_T g_max_value=0;

    if(N>0)
    {
        for(int i=0; i<outer_loop_limit; i++)
        {
            int max_index;
            TYPE_T max_value=0;

            #pragma unroll
            for(int j=0; j < W ;j++)
            {
                 x[j]=read_channel_intel(CHANNEL_VECTOR_X);
                 if(fabs(x[j])>max_value)
                 {
                     max_index=i*W+j;
                     max_value=fabs(x[j]);
                 }
            }

            if(max_value>g_max_value)
            {
                g_max_index=max_index;
                g_max_value=max_value;
            }

        }
    }
    write_channel_intel(CHANNEL_OUT,g_max_index);

}
