/*
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2020 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    SYR2  performs the symmetric rank 2 operation

    A := alpha*x*y**T + alpha*y*x**T + A,

    where alpha is a scalar, x and y are n element vectors and A is an n
    by n symmetric matrix.

    Data is received from five different channels ({{ channels["channel_in_vector_x"] }},
    {{ channels["channel_in_vector_x_trans"] }}, {{ channels["channel_in_vector_y"] }}, {{ channels["channel_in_vector_y_trans"] }} and {{ channels["channel_in_matrix_A"] }}).
    If A is an upper triangular matrix, it can arrives in tiles by row
    and Row Streamed. If A is a lower triangular matrix, it can arrives
    in tiles by cols and Col streamed.
    A is sent in packed format (that is, only the interesting elements are transmitted, padded to the Tile Size).

    Result is streamed in an output channel, tile by tile as soon as it is available.

    Check the kernel documentation for further information

*/


#pragma OPENCL EXTENSION cl_intel_channels : enable



{% if routine.type_str == 'double' %}
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
{% endif %}

channel {{ routine.type_str }} {{ channels["channel_in_vector_x"] }}  __attribute__((depth({{ routine.width }})));
channel {{ routine.type_str }} {{ channels["channel_in_vector_x_trans"] }}  __attribute__((depth({{ routine.width }})));
channel {{ routine.type_str }} {{ channels["channel_in_vector_y"] }}  __attribute__((depth({{ routine.width }})));
channel {{ routine.type_str }} {{ channels["channel_in_vector_y_trans"] }}  __attribute__((depth({{ routine.width }})));
channel {{ routine.type_str }} {{ channels["channel_in_matrix_A"] }}  __attribute__((depth({{ routine.width }})));
channel {{ routine.type_str }} {{ channels["channel_out_matrix"] }}  __attribute__((depth({{ routine.width }})));


/*
    - A is row streamed, tiles by rows, upper triangular
        - x and y are sent only once
        - x_trans and y_trans are sent multiple time in the following way:
            for each tile-row ti, we need blocks ti,....BlocksN as x_ and y_transposed
    - A is col streamed, tiles by cols, lower triangular
        - data  must arrive as in the previous case
*/
__kernel void {{ routine.user_name }}(const {{ routine.type_str }} alpha, const int N)
{
    __constant uint WIDTH = {{ routine.width }};
    __constant uint TILE_N = {{ routine.tile_n_size }};

    const int BlocksN=1+(int)((N-1)/TILE_N);
    const int reading_x_outer_loop_limit=(int)(TILE_N/WIDTH);

    {{ routine.type_str }} local_A[WIDTH];
    {{ routine.type_str }} local_x[TILE_N];
    {{ routine.type_str }} local_x_trans[TILE_N];
    {{ routine.type_str }} local_y[TILE_N];
    {{ routine.type_str }} local_y_trans[TILE_N];

    /**
        For each tile row ti
        - we need the ti-th block of x (and y)
        - we need the blocks ti,...,BlocksN of x (and y) (act as x transposed)
    */

    for(int ti=0; ti<BlocksN;ti++)
    {
        //here we reuse a block of x
        for(int i=0;i<reading_x_outer_loop_limit;i++)
        {
            #pragma unroll
            for(int ii=0;ii<WIDTH;ii++)
            {
                local_x[i*WIDTH+ii]=read_channel_intel({{ channels["channel_in_vector_x"] }});
                local_y[i*WIDTH+ii]=read_channel_intel({{ channels["channel_in_vector_y"] }});
            }
        }
        //operates on the lower part of the matrix
        for(int tj=ti;tj<BlocksN;tj++)
        {
            for(int i=0;i<reading_x_outer_loop_limit;i++) //buffer x transposed
            {
                #pragma unroll
                for(int ii=0;ii<WIDTH;ii++)
                {
                    local_x_trans[i*WIDTH+ii]=read_channel_intel({{ channels["channel_in_vector_x_trans"] }});
                    local_y_trans[i*WIDTH+ii]=read_channel_intel({{ channels["channel_in_vector_y_trans"] }});
                }
            }
            for(int i=0;i<TILE_N;i++)
            {
                const int i_idx=ti*TILE_N+i;
                //receive the row of A
                const int reading_A_limit=(tj>ti)?((int)(TILE_N/WIDTH)):ceilf(((float)(TILE_N-i))/WIDTH);
                const int empty_A_blocks=TILE_N/WIDTH-reading_A_limit; //in this case we receive only the meaningful part of a matrix
                for(int j=0;j<reading_A_limit;j++)
                {

                    {{ routine.type_str }} tmp1=alpha*local_x[i];
                    {{ routine.type_str }} tmp2=alpha*local_y[i];
                    #pragma unroll
                    for(int jj=0;jj<WIDTH;jj++)
                    {

                        //compute and send out the data
                        {{ routine.type_str }} a=read_channel_intel({{ channels["channel_in_matrix_A"] }});
                        local_A[jj]=tmp1*local_x_trans[empty_A_blocks*WIDTH+j*WIDTH+jj]+tmp2*local_y_trans[empty_A_blocks*WIDTH+j*WIDTH+jj]+a;

                        if(empty_A_blocks*WIDTH+tj*TILE_N+j*WIDTH+jj<i_idx)
                            local_A[jj]=0;
                        write_channel_intel({{ channels["channel_out_matrix"] }},local_A[jj]);
                    }
                }
            }
        }
    }
}
