/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2020 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    Reads a triangular upper matrix of type  {{ helper.type_str }} from memory and  push it
    into {{ channels["channel_out_matrix"] }}. The matrix is sent RowStreamed (i.e. one row after another)
    and Tiles are sent by row. Tiles have size TILE_N x TILE_N.

    The matrix is sent in packed format, i.e. for each row only the meaningful
    elements are sent, padded to WIDTH.

    {{ helper.width }} reads are performed simultaneously.
    If needed, data is padded to tile sizes using zero elements.
*/

__kernel void  {{ helper_name }}(__global  {{ helper.type_str }} *restrict data, int N,unsigned int lda)
{
    __constant uint WIDTH = {{ helper.width }};
    __constant uint TILE_N = {{ helper.tile_n_size }};
    __constant uint TILE_M = {{ helper.tile_m_size }};

    const int BlocksN=1+(int)((N-1)/TILE_N);


    for(int ti=0;ti<BlocksN;ti++)
    {
        for(int tj=ti;tj<BlocksN;tj++) //send only upper tiles
        {

            for(int i = 0; i < TILE_N; i++)
            {
                const int outer_loop_limit=(tj>ti)?((int)(TILE_N/WIDTH)):ceilf(((float)(TILE_N-i))/WIDTH);
                int i_idx=ti*TILE_N+i;

                for(int j=TILE_N/WIDTH-outer_loop_limit;j<TILE_N/WIDTH;j++)
                {
                    {{ helper.type_str }} to_send[WIDTH];
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
