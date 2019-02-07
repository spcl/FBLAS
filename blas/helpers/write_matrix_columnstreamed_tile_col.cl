/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2019 ETH-Zurich. All rights reserved.
    See LICENSE for license information.


    Writes a matrix of type TYPE_T in memory reading it
    from CHANNEL_MATRIX_OUT. The matrix is sent Column streamed (i.e. one column after another)
    and Tiles are sent by column. Tiles have size TILE_N x TILE_M.

    The name of the kernel can be redefined by means of preprocessor MACROS.
    Tile sizes must be defined by Macros.

    W reads are performed simultaneously.
    Padding data is discarded and not written in memory.

*/


__kernel void WRITE_MATRIX (__global volatile float *restrict matrix,int N, int M)
{
    const int BlocksN=N/TILE_N;
    const int BlocksM=M/TILE_M;
    int outer_loop_limit=(int)(TILE_N/W);


    for(int tj=0;tj<BlocksM;tj++)
    {
        for(int ti=0;ti<BlocksN;ti++)
        {
            for(int j = 0; j < TILE_M; j++)
            {
                for(int i=0;i<outer_loop_limit;i++)
                {
                    TYPE_T r[TILE_N];
                    #pragma unroll
                    for(int ii= 0; ii < W; ii++)
                        r[ii] = read_channel_intel(CHANNEL_MATRIX_OUT);

                    for(int ii=0; ii<W;ii++)
                        if((ti*TILE_N+i*W+ii)<N  && tj*TILE_N+j< M) //skip padding data
                            matrix[(ti*TILE_N+i*W+ii)*M+(tj*TILE_M+j)]=r[ii];

                }
            }
        }
    }
}
