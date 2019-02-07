/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2019 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    SCAL scales a vector by a constant.
    If the DOUBLE_PRECISION macro is defined, the operation is computed
    over double precision numbers, otherwise over single precision (float) numbers.

    Data is received through an input channel CHANNEL_VECTOR_X.
    Results are produced into the output channel CHANNEL_VECTOR_OUT.
    Data must arrive (and it is produced) padded with size W.
    Padding data must be set (or is set) to zero

*/



#pragma OPENCL EXTENSION cl_intel_channels : enable


//FBLAS_PARAMETERS_START

//#define DOUBLE_PRECISION      //Enable if double precision (dscal)
#define W 32			//width

//namings
#define KERNEL_NAME streaming_scal

//channel definitions
#define CHANNEL_VECTOR_X channel_float_gen_x
#define CHANNEL_VECTOR_OUT channel_float_sink

//architecture
#define __STRATIX_10__

//FBLAS_PARAMETERS_END

#include <commons.h>

channel TYPE_T CHANNEL_VECTOR_X __attribute__((depth(W)));
channel TYPE_T CHANNEL_VECTOR_OUT __attribute__((depth(W)));



__kernel void KERNEL_NAME(unsigned int N, TYPE_T alpha)
{

    const int outer_loop_limit=1+(int)((N-1)/W); //ceiling
    TYPE_T x[W];

    for(int i=0; i<outer_loop_limit; i++)
    {

        #pragma unroll
        for(int j=0;j<W;j++)
        {
            x[j]=alpha*read_channel_intel(CHANNEL_VECTOR_X);
            write_channel_intel(CHANNEL_VECTOR_OUT,x[j]);
        }

    }

}
