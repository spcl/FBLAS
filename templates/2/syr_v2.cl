/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2020 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    SYR   performs the symmetric rank 1 operation

        A := alpha*x*x**T + A,

    where alpha is a real scalar, x is an n element vector and A is an
    n by n symmetric matrix.

    Data is received from three different channels ({{ channels["channel_in_vector_x"] }},
    {{ channels["channel_in_vector_x_trans"] }} and {{ channels["channel_in_matrix_A"] }}).
    If A is an upper triangular matrix, it can arrives in tiles by row
    and Row Streamed. If A is a lower triangular matrix, it can arrives
    in tiles by cols and Col streamed.
    A is sent in packed format  (that is, only the interesting elements are transmitted, padded to the Tile Size).

    Result is streamed in an output channel, tile by tile as soon as it is available.

    Check the kernel documentation for further information

*/

#pragma OPENCL EXTENSION cl_intel_channels : enable
{% if routine.type_str == 'double' %}
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
{% endif %}

channel {{ routine.type_str }} {{ channels["channel_in_vector_x"] }}  __attribute__((depth({{ routine.width }})));
channel {{ routine.type_str }} {{ channels["channel_in_vector_x_trans"] }}  __attribute__((depth({{ routine.width }})));
channel {{ routine.type_str }} {{ channels["channel_in_matrix_A"] }}  __attribute__((depth({{ routine.width }})));
channel {{ routine.type_str }} {{ channels["channel_out_matrix"] }}  __attribute__((depth({{ routine.width }})));


/*
    In this version:
    - A is row streamed, tiles by rows, upper triangular
    - or A is column streamed, tiles by cols, lower triangular

    In both cases, it receives from two streams the vector x. One of the
    acts as x transposed. The trasposed x is sent multiple time: at first iterations
    it arrives block 0 to blocksN,  then block1 to N, ....

*/
__kernel void {{ routine.user_name }}(const {{ routine.type_str }} alpha, const int N)
{
    __constant uint WIDTH = {{ routine.width }};
    __constant uint TILE_N = {{ routine.tile_n_size }};

    //loops for buffering x and y are unrolled
    const int reading_x_outer_loop_limit=(int)(TILE_N/WIDTH);
    const int BlocksN=1+(int)((N-1)/TILE_N);

    {{ routine.type_str }} local_A[WIDTH];
    {{ routine.type_str }} local_x[TILE_N];
    {{ routine.type_str }} local_x_trans[TILE_N];

    /**
        For each tile row ti
        - we need the ti-th block of x
        - we need the blocks ti,...,BlocksN of x (act as x transposed)
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
            }
        }
        //operates on the upper part of the matrix
        for(int tj=ti;tj<BlocksN;tj++)
        {
            for(int i=0;i<reading_x_outer_loop_limit;i++) //buffer x transposed
            {
                #pragma unroll
                for(int ii=0;ii<WIDTH;ii++)
                {
                    local_x_trans[i*WIDTH+ii]=read_channel_intel({{ channels["channel_in_vector_x_trans"] }});
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

                    {{ routine.type_str }} tmp=alpha*local_x[i];
                    #pragma unroll
                    for(int jj=0;jj<WIDTH;jj++)
                    {

                        //compute and send out the data
                        {{ routine.type_str }} a=read_channel_intel({{ channels["channel_in_matrix_A"] }});
                        local_A[jj]=tmp*local_x_trans[empty_A_blocks*WIDTH+j*WIDTH+jj]+a;

                        //pad data
                        if(empty_A_blocks*WIDTH+tj*TILE_N+j*WIDTH+jj<i_idx)
                            local_A[jj]=0;
                        write_channel_intel({{ channels["channel_out_matrix"] }},local_A[jj]);
                    }
                }
            }
        }
    }
}


