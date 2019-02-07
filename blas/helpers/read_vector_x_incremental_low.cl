/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2019 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    Reads a vector of type TYPE_T from memory and  push it
    into CHANNEL_VECTOR_X_TRANS. The vector is accessed with stride INCX.
    At first iterations it generates block 0, then block 0 and 1, ....
    Block size is given by macro TILE_N.

    The name of the kernel can be redefined by means of preprocessor macro READ_VECTOR_X_TRANS.

    W memory reads are performed simultaneously. In the same way W channel pushes are performed .
    Data is padded to TILE_N using zero elements.

    It is used for routines SYR and SYR2
*/


__kernel void READ_VECTOR_X_TRANS(__global TYPE_T *restrict data, unsigned int N)
{
    const int BlocksN=1+(int)((N-1)/TILE_N);
    int outer_loop_limit=(int)(TILE_N/W);
    TYPE_T x[W];
    for(int ti=0; ti<BlocksN; ti++)
    {
        int offset=((INCX) > 0 ?  0 : ((N) - 1) * (-(INCX)));
        //send all the previous blocks
        //plus the current
        for(int tj=0;tj<=ti;tj++)
        {
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
                    write_channel_intel(CHANNEL_VECTOR_X_TRANS,x[k]);
            }
        }
    }
}
