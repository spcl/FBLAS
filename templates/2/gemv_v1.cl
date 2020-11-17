/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2020 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    GEMV_V1  performs one of the matrix-vector operations

    -   y := alpha*A*x + beta*y,
            where the NxM matrix A is received in tiles streamed by rows, where each
            tile is Row Streamed. x is an M-elements vector, while y
            is an N-element vector

    -   or  y := alpha*A**T*x + beta*y,
            where the NxM matrix A is received in tiles streamed by columns,
            each tile is Column Streamed. x is an N-element vector, while y
            is an M-element vector

    Data is received from three different channels ({{ channels["channel_in_vector_x"] }}, {{ channels["channel_in_vector_y"] }}
    and {{ channels["channel_in_matrix_A"] }}). Input data must be padded with zeros according to
    the reference tile sizes (TILE_N and TILE_M).

    Result is streamed in an output channel as soon as it is available.

    Check the kernel documentation for further information

*/

#pragma OPENCL EXTENSION cl_intel_channels : enable

{% if routine.type_str == 'double' %}
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
{% endif %}


channel {{ routine.type_str }} {{ channels["channel_in_vector_x"] }} __attribute__((depth({{ routine.width }})));
channel {{ routine.type_str }} {{ channels["channel_in_vector_y"] }} __attribute__((depth({{ routine.width }})));
channel {{ routine.type_str }} {{ channels["channel_in_matrix_A"] }} __attribute__((depth({{ routine.width }})));
channel {{ routine.type_str }} {{ channels["channel_out_vector"] }} __attribute__((depth(1)));

/**
    This version is meant for the following cases:
    - A is RowStreamed and NonTransposed, Tiles received by rows. In this case:
            - row_streamed must be set to 1
            - x is a vector of M elements, y is a vector of N elements
            - blocks of TILE_N elements of y are reused
            - also block of TILE_M elements of x are reused. The entire vector x must be resent N/TILE_N times (i.e. len_y/tile_y)
    - A is ColStreamed and Transposed, Tiles received by cols:
            - row_streamed must be set to 0
            - x is a vector of N elements, while y is a vector of M elements
            - blocks of y are of TILE_M elements
            - the entire x must be resent M/TILE_M times. Reuse will be applied also to it

    Matrix and vector must be padded to the respective tiling sizes
*/
__kernel void {{ routine.user_name }}(int row_streamed, const int N, const int M, const {{ routine.type_str }} alpha, const {{ routine.type_str }} beta)
{
    __constant uint WIDTH = {{ routine.width }};
    __constant uint TILE_N = {{ routine.tile_n_size }};
    __constant uint TILE_M = {{ routine.tile_m_size }};
    {% if routine.tile_n_size > routine.tile_m_size %}
    __constant uint MAX_TILE_SIZE = {{ routine.tile_n_size }};
    {% else %}
    __constant uint MAX_TILE_SIZE = {{ routine.tile_m_size }};
    {% endif %}

    {% if routine.uses_shift_registers %}
    __constant uint SHIFT_REG = {{ routine.size_shift_registers }};
    {% endif %}

    int len_x,tile_x;
    int len_y,tile_y;
    int BlocksX, BlocksY;
    //chose the loop limits
    if(row_streamed == 1)
    {
        len_x = M;
        len_y = N;
        tile_x=TILE_M;
        tile_y=TILE_N;
        BlocksY=1+(int)((N-1)/TILE_N); //ceiling
        BlocksX=1+(int)((M-1)/TILE_M);
    }
    else
    {	//in this case A is transposed
        len_x = N;
        len_y = M;
        tile_x=TILE_N;
        tile_y=TILE_M;
        BlocksY=1+(int)((M-1)/TILE_M);
        BlocksX=1+(int)((N-1)/TILE_N);
    }

    {% if routine.uses_shift_registers %}
    {{ routine.type_str }} shift_reg[SHIFT_REG+1]; //shift register

    #pragma unroll
    for(int i=0;i<SHIFT_REG+1;i++)
       shift_reg[i]=0;
    {% endif %}


    //The computation is performed by receiving A in tiles by row (A non transposed) or column (A transposed).
    //In this way, the result is computed by 'accumulating' over y elements
    //One block of y is computed for each row-tile (or column-tile) of A and using the entire x

    const int computing_outer_loop_limit=(int)(tile_x/WIDTH);
    const int reading_y_outer_loop_limit=(int)(tile_y/WIDTH);

    {{ routine.type_str }} local_y[MAX_TILE_SIZE];
    {{ routine.type_str }} local_x[MAX_TILE_SIZE];

    //Please note: the order in which tiles arrive, will determine the computation
    //(i.e. do not assume that you will receive the tiles one row after the other...maybe they can arrive column by column)

    #pragma loop_coalesce
    #pragma ivdep
    for(int ti=0;ti<BlocksY;ti++)
    {
        #pragma ivdep
        for(int tj=0;tj<BlocksX;tj++)
        {
            //To buffer x, we will use the first iteration of the main loop
            //Also here, do not be confused by i and j, they can refer to rows and column of columns and rows

            #pragma ivdep
            for(int i=0;i<tile_y;i++)
            {

                {{ routine.type_str }} prev;

                //here we read one element from A and one element from X and we use it
                //For X we buffer it at the first iteration
                //this should not be a problem if tile_y is distant
                {{ routine.type_str }} acc_o=0;

                #pragma unroll
                for(int i=0;i<SHIFT_REG+1;i++)
                    shift_reg[i]=0;

                #pragma ivdep
                for(int jj=0;jj<computing_outer_loop_limit;jj++)
                {

                    if(tj==0 && jj==0)//put here to have evertyhing in the loop
                    {
                        if(beta==0)
                            prev=0;
                        else
                           prev=beta*read_channel_intel({{ channels["channel_in_vector_y"] }});
                    }

                    if(i==0)
                    #pragma unroll
                    for(int j=0;j<WIDTH;j++)
                           local_x[jj*WIDTH+j]=read_channel_intel({{ channels["channel_in_vector_x"] }});

                    {{ routine.type_str }} acc_i=0;
                    //read (a block of W elements) of the row of A
                    {{ routine.type_str }} local_A[WIDTH];
                    #pragma unroll
                    for(int j=0;j<WIDTH;j++)
                        local_A[j]=read_channel_intel({{ channels["channel_in_matrix_A"] }});

                    #pragma unroll
                    for(int j=0;j<WIDTH;j++)
                        acc_i+=local_A[j]*local_x[jj*WIDTH+j];

                    shift_reg[SHIFT_REG] = shift_reg[0]+alpha*acc_i;
                    //Shift every element of shift register
                    #pragma unroll
                    for(int j = 0; j < SHIFT_REG; ++j)
                        shift_reg[j] = shift_reg[j + 1];

                    acc_o=0;
                    #pragma unroll
                    for(int i=0;i<SHIFT_REG;i++)
                    {
                        acc_o+=shift_reg[i];
                    }
                    if(jj == computing_outer_loop_limit -1){
                    //         //no beta version
                        if(tj!=0)
                            prev=local_y[i];
                        {{ routine.type_str }} result =  prev+  acc_o;
                        local_y[i] = result;
                        //output y if we reached the end of the matrix
                        //y is output one element at a time
                        if(tj==BlocksX-1)
                          write_channel_intel({{ channels["channel_out_vector"] }},result);
                    }
                }

            }
        }
    }
}
