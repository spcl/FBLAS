/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2019 ETH-Zurich. All rights reserved.
    See LICENSE for license information.


    Write a vector of type TYPE_T into  memory.
    The vector elements are read from channel CHANNEL_VECTOR_OUT.
    The name of the kernel can be redefined by means of preprocessor MACROS.
    INCW represent the access stride.

    W reads are performed simultaneously.
    Data arrives padded at pad_size.
    Padding data (if present) is discarded.
*/

__kernel void WRITE_VECTOR(__global TYPE_T *restrict out, unsigned int N,unsigned int pad_size)
{
    const unsigned int ratio=pad_size/W;
    const unsigned int padding_loop_limit=ceil(((float)N)/pad_size);
    const unsigned int outer_loop_limit=padding_loop_limit*ratio;
    TYPE_T recv[W];
    //compute the starting index
    int offset=((INCW) > 0 ?  0 : ((N) - 1) * (-(INCW)));
    //receive and store data into memory
    for(int i=0;i<outer_loop_limit;i++)
    {
        #pragma unroll
        for(int j=0;j<W;j++)
        {
            recv[j]=read_channel_intel(CHANNEL_VECTOR_OUT);

            if(i*W+j<N)
                out[offset+(j*INCW)]=recv[j];
        }
        offset+=W*INCW;
    }
}
