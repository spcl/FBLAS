/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2019 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    Reads a triangular upper matrix of type TYPE_T from memory and  push it
    into CHANNEL_MATRIX_A. The matrix is sent RowStreamed (i.e. one row after another)
    and Tiles are sent by row. Tiles have size TILE_N x TILE_N.

    The matrix is sent in packed format, i.e. for each row only the meaningful
    elements are sent, padded to W.

    The name of the kernel can be redefined by means of preprocessor MACROS.
    Tile sizes must be defined by Macros.

    W reads are performed simultaneously.
    If needed, data is padded to tile sizes using zero elements.
*/

__kernel void READ_MATRIX_A(__global TYPE_T *restrict data, int N,unsigned int lda)
{
    const int BlocksN=1+(int)((N-1)/TILE_N);

    TYPE_T to_send[W];

    for(int ti=0;ti<BlocksN;ti++)
    {
        for(int tj=ti;tj<BlocksN;tj++) //send only upper tiles
        {

            for(int i = 0; i < TILE_N; i++)
            {
                const int outer_loop_limit=(tj>ti)?((int)(TILE_N/W)):ceilf(((float)(TILE_N-i))/W);
                int i_idx=ti*TILE_N+i;

                for(int j=TILE_N/W-outer_loop_limit;j<TILE_N/W;j++)
                {
                    #pragma unroll
                    for(int jj=0;jj<W;jj++)
                    {
                        int j_idx=tj*TILE_N+j*W+jj;
                        if(j_idx>=i_idx && i_idx < N && j_idx < N)
                            to_send[jj]=data[i_idx*lda+j_idx];
                        else
                            to_send[jj]=0;
                    }

                    #pragma unroll
                    for(int jj = 0; jj < W; jj++)
                        write_channel_intel(CHANNEL_MATRIX_A,to_send[jj]);

                }
            }
        }
    }
}
