/**

    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2020 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    Reads a matrix of type {{ helper.type_str }} from memory and  push it
    into {{ channels["channel_out_matrix"] }}. The matrix is sent RowStreamed (i.e. one row after another)
    and Tiles are sent by Column. Tiles have size {{ helper.tile_n_size }} x {{ helper.tile_m_size }}.


    W reads are performed simultaneously.
    If needed, data is padded to tile sizes using zero elements.

*/

{% if generate_channel_declaration is defined %}
#pragma OPENCL EXTENSION cl_intel_channels : enable

channel {{ helper.type_str }} {{ channels["channel_out_matrix"] }} __attribute__((depth({{ helper.width }})));
{% endif %}

__kernel void {{ helper_name }}(__global volatile {{ helper.type_str }} *restrict data, int N, int M, unsigned int lda)
{
    __constant uint WIDTH = {{ helper.width }};
    __constant uint TILE_N = {{ helper.tile_n_size }};
    __constant uint TILE_M = {{ helper.tile_m_size }};

    const int BlocksN=1+((int)((N-1)/TILE_N));
    const int BlocksM=1+((int)((M-1)/TILE_M));

    const int outer_loop_limit=((int)TILE_M)/WIDTH;

    #pragma loop_coalesce
    for(int tj=0;tj<BlocksM;tj++)
    {
        for(int ti=0;ti<BlocksN;ti++)
        {
            for(int i = 0; i < TILE_N; i++)
            {
                for(int j=0; j < outer_loop_limit; j++ )
                {
                    {{ helper.type_str }} to_send[WIDTH];
                    #pragma unroll
                    for(int jj = 0; jj < WIDTH; jj++)
                    {
                        if((ti*TILE_N+i)<N  && tj*TILE_M+j*WIDTH+jj< M)
                            to_send[jj] = data[(ti*TILE_N+i)*lda + tj* TILE_M+j*WIDTH+jj];
                        else
                            to_send[jj]=0; //padding
                    }

                    #pragma unroll
                    for(int jj = 0; jj < WIDTH; jj++)
                        write_channel_intel({{ channels["channel_out_matrix"] }},to_send[jj]);
                }
            }
        }
    }
}
