/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2019 ETH-Zurich. All rights reserved.
    See LICENSE for license information.


    Writes a matrix of type TYPE_T in memory reading it
    from CHANNEL_MATRIX_OUT. The matrix is sent RowStreamed (i.e. one row after another)
    and Tiles are sent by row. Tiles have size TILE_N x TILE_M.

    The name of the kernel can be redefined by means of preprocessor MACROS.
    Tile sizes must be defined by Macros.

    W reads are performed simultaneously.
    Padding data is discarded and not written in memory.

*/


__kernel void WRITE_MATRIX (__global TYPE_T *restrict matrix, int N, int M, unsigned int lda)
{
    const int BlocksN=1+(int)((N-1)/TILE_N);
    const int BlocksM=1+(int)((M-1)/TILE_M);
    int outer_loop_limit=(int)(TILE_M/W);


    for(int ti=0;ti<BlocksN;ti++)
    {
        for(int tj=0;tj<BlocksM;tj++)
        {
            for(int i = 0; i < TILE_N; i++)
            {
                for(int j=0;j<outer_loop_limit;j++)
                {
                    TYPE_T r[TILE_M];
                    #pragma unroll
                    for(int jj= 0; jj < W; jj++)
                        r[jj] = read_channel_intel(CHANNEL_MATRIX_OUT);

                    #pragma unroll
                    for(int jj= 0; jj < W; jj++)
                        if((ti*TILE_N+i)<N  && tj*TILE_N+j*W+jj< M) //skip padding data
                            matrix[(ti*TILE_N+i)*lda+(tj*TILE_M+j*W+jj)]=r[jj];
                }
            }
        }
    }
}
