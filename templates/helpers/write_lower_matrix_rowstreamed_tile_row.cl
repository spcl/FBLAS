/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2020 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    Writes a triangular lower matrix of type {{ helper.type_str }} in memory reading it
    from {{ channels["channel_in_matrix"] }}. The matrix is sent RowStreamed (i.e. one row after another)
    and Tiles are sent by row. Tiles have size TILE_N x TILE_N.

    Data arrives in the packed format, i.e. only the meaningful elements
    arrives (padded to WIDTH)

    {{ helper.width }} reads are performed simultaneously.
    Padding data is discarded and not written in memory.

*/

__kernel void {{ helper_name }}(__global {{ helper.type_str }} *restrict matrix,int N,unsigned int lda)
{

    __constant uint WIDTH = {{ helper.width }};
    __constant uint TILE_N = {{ helper.tile_n_size }};
    __constant uint TILE_M = {{ helper.tile_m_size }};
    const int BlocksN=1+(int)((N-1)/TILE_N);


    for(int ti=0;ti<BlocksN;ti++)
    {
        for(int tj=0;tj<=ti;tj++)
        {

            for(int i = 0; i < TILE_N; i++)
            {
                const int outer_loop_limit=(tj<ti)?((int)(TILE_N/WIDTH)):ceilf(((float)(i+1))/WIDTH);
                int i_idx=ti*TILE_N+i;
                for(int j=0;j<outer_loop_limit;j++)
                {
                    {{ helper.type_str }} r[WIDTH];
                    #pragma unroll
                    for(int jj= 0; jj < WIDTH; jj++)
                        r[jj] = read_channel_intel({{ channels["channel_in_matrix"] }});

                    #pragma unroll
                    for(int jj= 0; jj < WIDTH; jj++)
                    {
                        int j_idx=tj*TILE_N+j*WIDTH+jj;
                        if(j_idx<=i_idx && i_idx < N && j_idx < N) //we can write it
                            matrix[i_idx*N+(j_idx)]=r[jj];
                    }
                }
            }
        }
    }
}
