/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2020 ETH-Zurich. All rights reserved.
    See LICENSE for license information.


    TRSV  solves one of the systems of equations

       A*x = b,   or   A**T*x = b,

    where b and x are n element vectors and A is an n by n unit, or
    non-unit, upper or lower triangular matrix.

    No test for singularity or near-singularity is included in this
    routine.


    Data is received from two different channels, {{ channels["channel_in_vector_x"] }} and
    {{ channels["channel_in_matrix_A"] }}. At the first iteration we receive the vector b
    from {{ channels["channel_in_vector_x"] }}. The updates for vector x are sent through channel
    {{ channels["channel_out_vector"] }}.

    This routine version for the following cases.
    If A is an upper triangular matrix, it must arrive in tiles by row
    and Row Streamed, in the reverse order.
    If A is a lower triangular matrix, it must arrive in tiles by cols
    and Col streamed, in the reverse order.
    A is sent in packed format  (that is, only the interesting elements are transmitted, padded to the Tile Size).

    Check the kernel documentation for further information.

*/
#pragma OPENCL EXTENSION cl_intel_channels : enable


{% if routine.type_str == 'double' %}
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
{% endif %}


channel {{ routine.type_str }} {{ channels["channel_in_vector_x"] }}  __attribute__((depth({{ routine.tile_n_size }})));
channel {{ routine.type_str }} {{ channels["channel_in_matrix_A"] }}   __attribute__((depth({{ routine.width }})));
channel {{ routine.type_str }} {{ channels["channel_out_vector"] }} __attribute__((depth({{ routine.tile_n_size }})));


/**
    In this case:
    - A is row streamed, tiles by columns, A is an upper matrix and the computation is on A transposed
        -in this case for each row-tile TI of A, we receive the blocks
                form 0 to TI of x. At the end of the row-tile the block TI
                of x is updated
    - A is column streamed, tiles by rows, A is a lower matrix and the computation is one A non-transposed

*/
__kernel void {{ routine.user_name }}(int N)
{
    __constant uint WIDTH = {{ routine.width }};
    __constant uint TILE_N = {{ routine.tile_n_size }};
    const int BlocksN=1+(int)((N-1)/TILE_N);
    const int computing_outer_loop_limit=(int)(TILE_N/WIDTH);

    //in the following we will refer to y as the output and x the input (even if we are essentially overwriting x)
    {{ routine.type_str }} local_y[TILE_N];
    for(int i=0;i<TILE_N;i++)
        local_y[i]=0;

    /*
        Tiles arrives by columns
        Suppose that we are in the first case: since we are working on A transposed, for each row of
        the tile we will update all the corresponding y
    */


    for(int tj=0;tj<BlocksN;tj++)
    {

        for(int ti=0; ti<=tj;ti++)  //upper diagonal blocks
        {

            {{ routine.type_str }} local_x[TILE_N];
            for(int j=0;j<computing_outer_loop_limit;j++)
            {
                #pragma unroll
                for(int jj=0;jj<WIDTH;jj++)
                    local_x[j*WIDTH+jj]=read_channel_intel({{ channels["channel_in_vector_x"] }});
            }

            for(int i=0;i<TILE_N;i++)
            {
                {{ routine.type_str }} local_A[TILE_N];
                //receive the row of A (partially if we are on a diagonal tile)
                const int reading_A_limit=(tj>ti)?((int)(TILE_N)/WIDTH):ceilf(((float)(TILE_N-i))/WIDTH);
                const int empty_A_blocks=TILE_N/WIDTH-reading_A_limit;
                //even if we don't read them, place the 0 in the row of A
                for(int j=0;j<reading_A_limit;j++)
                {
                    #pragma unroll
                    for(int jj=0;jj<WIDTH;jj++)
                        local_A[j*WIDTH+jj]=read_channel_intel({{ channels["channel_in_matrix_A"] }});
                }


                if(tj==ti)
                {
                    //we are in a diagonal tile: at each row we are computing a new y[i]
                    local_y[i]+=local_x[i];
                    //divide for the diagonal element (we have to find it)
                    int remainder=((int)(TILE_N-i)%WIDTH);
                    int diag_idx=(remainder==0)? 0: WIDTH-remainder;
                    local_y[i]/=local_A[diag_idx];
                }

                #pragma ivdep array(local_y)	//no dependencies over y
                for(int j=0;j<computing_outer_loop_limit;j++)
                {
                    #pragma unroll
                    for(int jj=0;jj<WIDTH;jj++)
                    {
                        if(tj>ti)
                        {
                            //dense tile
                            local_y[j*WIDTH+jj]-=local_A[j*WIDTH+jj]*local_x[i]; //use the value calculated at the beginning of this row tile
                        }
                        else
                        {
                            if(j*WIDTH+jj>i)
                            {
                                //in this case, the incoming tile of A is triangular
                                //since we are receiving the matrix in a packed format, local_A is not full, we have to skip the empty pieces
                                local_y[j*WIDTH+jj]-=local_A[j*WIDTH+jj-WIDTH*empty_A_blocks]*local_y[i];
                            }
                        }
                    }
                }
            }
        }
        //we can send the update
        for(int i=0;i<computing_outer_loop_limit;i++)
        {
            #pragma unroll
            for(int j=0;j<WIDTH;j++)
            {
                write_channel_intel({{ channels["channel_out_vector"] }},local_y[i*WIDTH+j]);
                local_y[i*WIDTH+j]=0;
            }
        }
    }
}
