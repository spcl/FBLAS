/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2020 ETH-Zurich. All rights reserved.
    See LICENSE for license information.


    Writes a matrix of type {{ helper.type_str }} in memory reading it
    from {{ channels["channel_in_matrix"] }}. The matrix is sent RowStreamed (i.e. one row after another)
    and Tiles are sent by row. Tiles have size {{ helper.tile_n_size }} x {{ helper.tile_m_size }}.

    {{ helper.width }} reads are performed simultaneously.
    Padding data is discarded and not written in memory.

*/
{% if generate_channel_declaration is defined %}
#pragma OPENCL EXTENSION cl_intel_channels : enable

channel {{ helper.type_str }} {{ channels["channel_in_matrix"] }} __attribute__((depth({{ helper.width }})));
{% endif %}

__kernel void {{ helper_name }}(__global volatile {{ helper.type_str }} *restrict matrix, int N, int M, unsigned int lda)
{
    __constant uint WIDTH = {{ helper.width }};
    __constant uint TILE_N = {{ helper.tile_n_size }};
    __constant uint TILE_M = {{ helper.tile_m_size }};

    const int BlocksN=1+(int)((N-1)/TILE_N);
    const int BlocksM=1+(int)((M-1)/TILE_M);
    int outer_loop_limit=(int)(TILE_M/WIDTH);

    #pragma loop_coalesce
    for(int ti=0;ti<BlocksN;ti++)
    {
        for(int tj=0;tj<BlocksM;tj++)
        {
            for(int i = 0; i < TILE_N; i++)
            {
                for(int j=0;j<outer_loop_limit;j++)
                {
                    {{ helper.type_str }} r[TILE_M];
                    #pragma unroll
                    for(int jj= 0; jj < WIDTH; jj++)
                        r[jj] = read_channel_intel({{ channels["channel_in_matrix"] }});

                    #pragma unroll
                    for(int jj= 0; jj < WIDTH; jj++)
                        if((ti*TILE_N+i)<N  && tj*TILE_N+j*WIDTH+jj< M) //skip padding data
                            matrix[(ti*TILE_N+i)*lda+(tj*TILE_M+j*WIDTH+jj)]=r[jj];
                }
            }
        }
    }
}
