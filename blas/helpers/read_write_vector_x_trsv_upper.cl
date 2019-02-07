/*
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2019 ETH-Zurich. All rights reserved.
    See LICENSE for license information.


    This helper is intended to be used with trsv

    Reads a vector of type TYPE_T from memory and  push it
    into CHANNEL_VECTOR_X.
    The vector is sent in blocks and each block may be sent multiple times.
    Blocs are produced in the reverse order.
    Being BN the number of blocks of the vector x, at the iteration i
    it will sends the blocks from BN-1 to i.
    Inside a block, elements are produced in order.
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
    for(int ti=BlocksN-1; ti>=0; ti--)
    {
        //send all the previous blocks
        //plus this one that we will receive back
        //we start from the bottom

        for(int tj=BlocksN-1;tj>=ti;tj--)
        {
            int offset=((INCX) > 0 ?  0 : ((N) - 1) * (-(INCX)))+ INCX * tj*TILE_N;

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
        int offset=((INCX) > 0 ?  0 : ((N) - 1) * (-(INCX)))+ INCX * ti*TILE_N;
        for(int i=0;i<outer_loop_limit;i++)
        {

            #pragma unroll
            for(int ii=0; ii<W ;ii++) //skip padding data
            {
                float r=read_channel_intel(CHANNEL_VECTOR_OUT);
                if(ti*TILE_N+i*W+ii<N)
                    data[offset+(ii*INCX)]=r;
            }
            offset+=W*INCX;

        }
    }
}

