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
    {{ channels["channel_in_matrix_A"] }} . At the first iteration we receive the vector b
    from {{ channels["channel_in_vector_x"] }}. The updates for vector x are sent through channel
    {{ channels["channel_out_vector"] }}.

    If A is a triangular lower matrix, it must arrive in tiles by row
    and Row Streamed. If A is an upper triangular matrix, it must arrive
    in tiles by cols and Col streamed.
    A is sent in packed format (that is, only the interesting elements are transmitted, padded to the Tile Size).

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
    This version can be used for the cases in which:
    - A is row streamed, tiles by row, non transposed and lower triangular:
        - in this case for each row-tile TI of A, we receive the blocks
            form 0 to TI of x. At the end of the row-tile the block TI
            of x is updated

    - A is col streamed, tiles by col. A is transposed and upper triangular
        - x is received/updated like the previous case

    The matrix is received in the packed format
    */

__kernel void {{ routine.user_name }}(int N)
{
    __constant uint WIDTH = {{ routine.width }};
    __constant uint TILE_N = {{ routine.tile_n_size }};
    {% if routine.uses_shift_registers %}
    __constant uint SHIFT_REG = {{ routine.size_shift_registers }};
    {% endif %}

    const int BlocksN=1+(int)((N-1)/TILE_N);
    const int computing_outer_loop_limit=(int)(TILE_N/WIDTH);

    {% if routine.uses_shift_registers %}
    {{ routine.type_str }} shift_reg[SHIFT_REG+1]; //shift register

    #pragma unroll
    for(int i=0;i<SHIFT_REG+1;i++)
       shift_reg[i]=0;
    {% endif %}
    

    //in the following we will refer to y as the output and x the input
    {{ routine.type_str }} local_y[TILE_N];
    {{ routine.type_str }} local_x[TILE_N];
    {{ routine.type_str }} local_A[TILE_N];
    for(int i=0;i<TILE_N;i++)
        local_y[i]=0;
    //for each iteration of this outer loop we will update
    //a block (of size TILE_N) of vector x
    for(int ti=0;ti<BlocksN;ti++)
    {
        for(int tj=0; tj<=ti;tj++)  //lower diagonal blocks (if A is row streamed, non transposed)
        {
            //receive the tj-th block of x

            for(int j=0;j<computing_outer_loop_limit;j++)
            {
                #pragma unroll
                for(int jj=0;jj<WIDTH;jj++)
                {
                    local_x[j*WIDTH+jj]=read_channel_intel({{ channels["channel_in_vector_x"] }});
                }
            }
            for(int i=0;i<TILE_N;i++)
            {
                //receive the row of A
                const int reading_A_limit=(tj<ti)?((int)(TILE_N)/WIDTH):ceilf(((float)(i+1))/WIDTH);
                for(int j=0;j<reading_A_limit;j++)
                {
                    #pragma unroll
                    for(int jj=0;jj<WIDTH;jj++)
                    {
                        local_A[j*WIDTH+jj]=read_channel_intel({{ channels["channel_in_matrix_A"] }} );
                    }
                }

                //compute: if we are on a diagonal block
                //we are updating the corresponding x elements
                //The basic computation is
                //FOR I=0...N-1
                //  x_out[i]=x[i];
                //  FOR J=0 .... I-1
                //      x_out[i]-=A[i][j]x[j]
                //  x_out[i]/=A[i][i]

                {{ routine.type_str }} acc_i=0,acc_o=0;
                int diag_idx=0;
                for(int j=0;j<computing_outer_loop_limit;j++)
                {
                    acc_i=0;
                    #pragma unroll
                    for(int jj=0;jj<WIDTH;jj++)
                    {
                        if(tj<ti)
                            acc_i-=local_A[j*WIDTH+jj]*local_x[j*WIDTH+jj];
                        else
                        {
                            if(j*WIDTH+jj<i)    //valid element
                            {
                                acc_i-=local_A[j*WIDTH+jj]*local_y[j*WIDTH+jj];
                                diag_idx++;
                            }
                        }
                    }
                    {% if routine.uses_shift_registers %}
                        shift_reg[SHIFT_REG] = shift_reg[0]+acc_i;
                        //Shift every element of shift register
                        #pragma unroll
                        for(int j = 0; j < SHIFT_REG; ++j)
                            shift_reg[j] = shift_reg[j + 1];
                    {% else %}
                        acc_o+=acc_i;
                    {% endif %}
                }
                {% if routine.uses_shift_registers %}
                    //reconstruct the result using the partial results in shift register
                    #pragma unroll
                    for(int i=0;i<SHIFT_REG;i++)
                    {
                        acc_o+=shift_reg[i];
                        shift_reg[i]=0;
                    }
                {% endif %}
                local_y[i]+=acc_o;
                if(tj==ti)
                {
                    local_y[i]+=local_x[i];
                    local_y[i]/=local_A[diag_idx];
                }

            }
        }
        //send the updated block of x
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
