/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2020 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    This helper is intended to be used with TRSV

    Reads a triangular upper matrix of type {{ helper.type_str }} from memory and  push it
    into {{ channels["channel_out_matrix"] }}, starting from the bottom row-tile.
    For each row-tile, tiles are produced in the reverse order, from right
    to left.
    Given a particular row, elements are produced in the natural order, i.e.
    from left to right.

    The matrix is sent in packed format, i.e. for each row only the meaningful
    elements are sent, padded to W.

    WIDTH reads are performed simultaneously.
    If needed, data is padded to tile sizes using zero elements.


*/

{% if generate_channel_declaration is defined %}
#pragma OPENCL EXTENSION cl_intel_channels : enable

channel {{ helper.type_str }} {{ channels["channel_out_matrix"] }} __attribute__((depth({{ helper.width }})));
{% endif %}

__kernel void {{ helper_name }}(__global volatile {{ helper.type_str }} *restrict data, int N, unsigned int lda)
{
    __constant uint WIDTH = {{ helper.width }};
    __constant uint TILE_N = {{ helper.tile_n_size }};

    const int BlocksN=1+(int)((N-1)/TILE_N);

    for(int ti=BlocksN-1;ti>=0;ti--)
    {
        for(int tj=BlocksN-1;tj>=ti;tj--)
        {
            for(int i = TILE_N-1; i >=0; i--)
            {
                const int outer_loop_limit=(tj>ti)?((int)(TILE_N/WIDTH)):ceilf(((float)(TILE_N-i))/WIDTH);
                int i_idx=ti*TILE_N+i;

                for(int j=TILE_N/WIDTH-outer_loop_limit;j<TILE_N/WIDTH;j++)
                {
                    {{ helper.type_str }} to_send[WIDTH];
                    //prepare data
                    #pragma unroll
                    for(int jj=0;jj<WIDTH;jj++)
                    {
                        int j_idx=tj*TILE_N+j*WIDTH+jj;
                        if(j_idx>=i_idx && i_idx < N && j_idx < N)
                            to_send[jj]=data[i_idx*lda+j_idx];
                        else
                            to_send[jj]=0;
                    }

                    #pragma unroll
                    for(int jj = 0; jj < WIDTH; jj++)
                        write_channel_intel({{ channels["channel_out_matrix"] }},to_send[jj]);
                }
            }
        }
    }
}
