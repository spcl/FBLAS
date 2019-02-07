/*
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2019 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    This helper is intended to be used with trsv

    Reads a vector of type TYPE_T from memory and  push it
    into CHANNEL_VECTOR_X.
    The vector is sent in blocks and each block may be sent multiple times.
    At the first iteration it sends the first block. Then the first and the second, ...
    At iteration i, it sends blocks from O to i-1.
    At each iteration, after sending blocks i, it gets back the same block updated
    from CHANNEL_VECTOR_OUT

    The vector is accessed with stride INCX.
    Block size is given by macro TILE_N.

    The name of the kernel can be redefined by means of preprocessor macro READ_VECTOR_X_TRSV.

    W memory reads are performed simultaneously. In the same way W channel pushes are performed .
    Data is padded to TILE_N using zero elements.


*/
__kernel void READ_VECTOR_X_TRSV(__global TYPE_T *restrict data, int N)
{
    const int BlocksN=1+(int)((N-1)/TILE_N);
    int outer_loop_limit=(int)(TILE_N/W);

    TYPE_T x[W];

    for(int ti=0; ti<BlocksN; ti++)
    {
        //send all the previous blocks
        //plus this one that we will receive back
        int offset=((INCX) > 0 ?  0 : ((N) - 1) * (-(INCX)));

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

                #pragma unroll
                for(int ii=0; ii<W ;ii++)
                {
                    write_channel_intel(CHANNEL_VECTOR_X,x[ii]);
                }
            }
        }

       //get back the result and overwrite it
       offset -= TILE_N*INCX;
       for(int i=0;i<outer_loop_limit;i++)
       {
           #pragma unroll
           for(int ii=0; ii<W ;ii++)
           {
                TYPE_T r=read_channel_intel(CHANNEL_VECTOR_OUT);
                if(ti*TILE_N+i*W+ii<N)
                    data[offset+(ii*INCX)]=r;
           }
           offset+=W*INCX;
       }
    }

}
