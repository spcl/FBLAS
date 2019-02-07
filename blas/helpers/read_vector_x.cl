/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2019 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    Reads a vector of type TYPE_T from memory and  push it
    into CHANNEL_VECTOR_X. The vector is accessed with stride INCX.
    The name of the kernel can be redefined by means of preprocessor MACROS.

    W memory reads are performed simultaneously. In the same way W channel push are performed.
    Data is padded to pad_size. Pad_size must be a multiple of W.
    So, for level 1 routines pad_size will be probably equal to W.
    For level 2, in which you have to respect some tiling, it will be equal to a tile size.

    Data is padded using zero elements.

    The vector is sent 'repetitions' times
*/

__kernel void READ_VECTOR_X(__global TYPE_T *restrict data, unsigned int N, unsigned int pad_size, unsigned int repetitions)
{
    unsigned int ratio=pad_size/W;

    unsigned int padding_loop_limit=ceil(((float)N)/pad_size);
    unsigned int outer_loop_limit=padding_loop_limit*ratio;
    TYPE_T x[W];
    for(int t=0; t< repetitions;t++)
    {
        //compute the starting index
        int offset=((INCX) > 0 ?  0 : ((N) - 1) * (-(INCX)));

        for(int i=0;i<outer_loop_limit;i++)
        {
            //prepare data
            #pragma unroll
            for(int k=0;k<W;k++)
            {
                if(i*W+k<N)
                    x[k]=data[offset+(k*INCX)];
                else
                    x[k]=0;
            }
            offset+=W*INCX;

            //send data
            #pragma unroll
            for(int k=0;k<W;k++)
                write_channel_intel(CHANNEL_VECTOR_X,x[k]);
        }
    }
}
