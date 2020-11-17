/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2020 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    Generates a dummy matrix  of type {{ helper.type_str }} without reading from memory and  push it
    into {{ channels["channel_out_matrix"] }}.

    Note: this is independent from the given tiles/elements order

*/

{% if generate_channel_declaration is defined%}
#pragma OPENCL EXTENSION cl_intel_channels : enable

channel {{ helper.type_str }} {{ channels["channel_out_matrix"] }} __attribute__((depth({{ helper.width }})));
{% endif %}


__kernel void {{ helper_name }}(int N, int M)
{

    __constant uint WIDTH = {{ helper.width }};
    __constant uint TILE_N = {{ helper.tile_n_size }};
    __constant uint TILE_M = {{ helper.tile_m_size }};
    const int BlocksN=1+(int)((N-1)/TILE_N);
    const int BlocksM=1+(int)((M-1)/TILE_M);
    const int outer_loop_limit=((int)(TILE_M))/WIDTH;   //WIDTH must be a divisor of TILE
    {{ helper.type_str }} to_send[WIDTH];
    #pragma loop_coalesce
    for(int ti=0;ti<BlocksN;ti++)
    {
        for(int tj=0;tj<BlocksM;tj++)
        {
            for(int i = 0; i < TILE_N; i++)
            {
                const {{ helper.type_str }} to_send=({{ helper.type_str }})(ti*TILE_N+i);
                for(int j=0; j < outer_loop_limit; j++ )
                {
                    #pragma unroll
                    for(int jj = 0; jj < WIDTH; jj++)
                        write_channel_intel({{ channels["channel_out_matrix"] }},to_send);

                }
            }
        }
    }
}