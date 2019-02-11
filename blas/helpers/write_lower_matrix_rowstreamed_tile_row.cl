/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2019 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    Writes a triangular lower matrix of type TYPE_T in memory reading it
    from CHANNEL_MATRIX_OUT. The matrix is sent RowStreamed (i.e. one row after another)
    and Tiles are sent by row. Tiles have size TILE_N x TILE_N.

    Data arrives in the packed format, i.e. only the meaningful elements
    arrives (padded to W)

    The name of the kernel can be redefined by means of preprocessor MACROS.
    Tile sizes must be defined by Macros.

    W reads are performed simultaneously.
    Padding data is discarded and not written in memory.

*/

__kernel void WRITE_MATRIX(__global volatile TYPE_T *restrict matrix,int N,unsigned int lda)
{
    const int BlocksN=1+(int)((N-1)/TILE_N);


    for(int ti=0;ti<BlocksN;ti++)
    {
        for(int tj=0;tj<=ti;tj++)
        {

            for(int i = 0; i < TILE_N; i++)
            {
                const int outer_loop_limit=(tj<ti)?((int)(TILE_N/W)):ceilf(((float)(i+1))/W);
                int i_idx=ti*TILE_N+i;
                for(int j=0;j<outer_loop_limit;j++)
                {
                    TYPE_T r[W];
                    #pragma unroll
                    for(int jj= 0; jj < W; jj++)
                        r[jj] = read_channel_intel(CHANNEL_MATRIX_OUT);

                    #pragma unroll
                    for(int jj= 0; jj < W; jj++)
                    {
                        int j_idx=tj*TILE_N+j*W+jj;
                        if(j_idx<=i_idx && i_idx < N && j_idx < N) //we can write it
                            matrix[i_idx*N+(j_idx)]=r[jj];
                    }
                }
            }
        }
    }
}
