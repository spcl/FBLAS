/**

    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2020 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    Reads a matrix of type {{ helper.type_str }} from memory and  push it
    into {{ channels["channel_out_matrix"] }}. The matrix is sent RowStreamed (i.e. one row after another)
    and Tiles are sent by row. Tiles have size {{ helper.tile_n_size }} x {{ helper.tile_m_size }}.

    This is a slightly different version of the basic matrix reader
    To exploit full memory bandwidth, we have to manually interleave between
    4 DRAM modules (Intel Compiler doesn't do this automatically).

    Data must be properly accommodated in DRAM (see gemv example, under sample/cpu_comparison).

    {{ helper.width }} reads are performed simultaneously.

    NOTE: currently it does not support padding.
    If needed, data is padded to tile sizes using zero elements.

*/


{% if generate_channel_declaration is defined %}
#pragma OPENCL EXTENSION cl_intel_channels : enable

channel {{ helper.type_str }} {{ channels["channel_out_matrix"] }} __attribute__((depth({{ helper.width }})));
{% endif %}


__kernel void {{ helper_name }}(__global volatile  {{ helper.type_str }} *restrict data0,__global volatile  {{ helper.type_str }} *restrict data1,
                            __global volatile  {{ helper.type_str }} *restrict data2,__global volatile  {{ helper.type_str }} *restrict data3, int N, int M, unsigned int lda)
{
    __constant uint WIDTH = {{ helper.width }};
    __constant uint TILE_N = {{ helper.tile_n_size }};
    __constant uint TILE_M = {{ helper.tile_m_size }};
    const int BlocksN=1+(int)((N-1)/TILE_N);
    const int BlocksM=1+(int)((M-1)/TILE_M);
    const int loop_it=((int)(TILE_M))/WIDTH;   //W must be a divisor of TILE
    const int multiply_width=1+(int)((lda-1)/WIDTH); //lda must be a multiple of width, otherwise inefficient hw is generated for the load

    {{ helper.type_str }} to_send[WIDTH];
    #pragma loop_coalesce
    for(int ti=0;ti<BlocksN;ti++)
    {
        for(int tj=0;tj<BlocksM;tj++)
        {
            for(int i=0;i<TILE_N;i++)
            {
                for(int j=0;j<loop_it;j++)
                {
                    const int row_idx=ti*TILE_N+i;

                    //load from memory
                    #pragma unroll
                    for(int k=0;k<WIDTH/4;k++)
                            to_send[k]=data0[row_idx*WIDTH/4*multiply_width+tj*TILE_M/4+j*WIDTH/4+k];
                    #pragma unroll
                    for(int k=0;k<WIDTH/4;k++)
                            to_send[k+WIDTH/4]=data1[row_idx*WIDTH/4*multiply_width+tj*TILE_M/4+j*WIDTH/4+k];
                    #pragma unroll
                    for(int k=0;k<WIDTH/4;k++)
                            to_send[k+WIDTH/2]=data2[row_idx*WIDTH/4*multiply_width+tj*TILE_M/4+j*WIDTH/4+k];
                    #pragma unroll
                    for(int k=0;k<WIDTH/4;k++)
                            to_send[k+3*WIDTH/4]=data3[row_idx*WIDTH/4*multiply_width+tj*TILE_M/4+j*WIDTH/4+k];

                    #pragma unroll
                    for(int k = 0; k < WIDTH; k++)
                        write_channel_intel({{ channels["channel_out_matrix"] }},to_send[k]);
                }
            }
        }
    }
}