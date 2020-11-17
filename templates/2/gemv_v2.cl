/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2020 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    GEMV_V2  performs one of the matrix-vector operations

    -   y := alpha*A*x + beta*y,  
            where the NxM matrix A is received in tiles streamed by rows, where each 
            tile is ColumnStreamed. x is an M-elements vector, while y
            is an N-element vector

    -   or  y := alpha*A**T*x + beta*y,
            where the NxM matrix A is received in tiles streamed by columns,
            each tile is Row Streamed. x is an N-element vector, while y
            is an M-element vector

    Data is received from three different channels ({{ channels["channel_in_vector_x"] }}, {{ channels["channel_in_vector_y"] }}
    and {{ channels["channel_in_matrix_A"] }} A). Input data must be padded with zeros according to
    the reference tile sizes ({{ routine.tile_n_size }} and {{ routine.tile_m_size }}).

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
channel {{ routine.type_str }} {{ channels["channel_out_vector"] }} __attribute__((depth({{ routine.width }})));


/**
    This version is meant for the following cases:
    - A is rowStreamed and Transposed. A is received by tiles by columns. For this case:
        - row_streamed must be set to 1
	- vector x is composed by N elements, y is composed by M elements
	- blocks of y are composed by Tile_M elements and they will be reused
	- blocks of x are composed by Tile_N elements, they will be not reused
	- x must be re-sent entirely M/TILE_M times

    - A is columnStreamed and Not Transposed. A is received by tiles in row ordering:
        - row_streamed must be set to 0
	- vector x is composed by M elements, vector y is composed by N elements
	- block of y are composed by TILE_N elements and they will be reused
	- block of x are composed by TILE_N elemenets, not reused
	- x must be re-sent N/TILE_N times
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

    int len_x,tile_x;
    int len_y,tile_y;
    int BlocksY, BlocksX;
    //chose the loop limits
    if(row_streamed==1)
    {
        len_x = N;
        len_y = M;
        tile_x=TILE_N;
        tile_y=TILE_M;
        BlocksY=1 + (int)((M-1)/TILE_M);
        BlocksX=1 + (int)((N-1)/TILE_N);
    }
    else
    {	//in this case A is non transposed
        len_x = M;
        len_y = N;
        tile_x=TILE_M;
        tile_y=TILE_N;
        BlocksY=1 + (int)((N-1)/TILE_N);
        BlocksX=1 + (int)((M-1)/TILE_M);
    }


    //In this case each element of x will be multiplied for all the tile_y elements of A
    const int computing_outer_loop_limit=(int)(tile_y/WIDTH);
    const int reading_y_outer_loop_limit=(int)(tile_y/WIDTH);


    for(int ti=0;ti<BlocksY;ti++)
    {
        //Reuse over y
        {{ routine.type_str }} local_y[MAX_TILE_SIZE];

        for(int i=0;i<reading_y_outer_loop_limit;i++)
        {
            if(beta == 0)
            {	//if beta is equal to zero we don't need to read from CHANNEL_Y
                #pragma unroll
                for(int j=0;j<WIDTH;j++)
                    local_y[i*WIDTH+j] = 0;
            }
            else
            {
                #pragma unroll
                for(int j=0;j<WIDTH;j++)
                    local_y[i*WIDTH+j]=beta*read_channel_intel({{ channels["channel_in_vector_y"] }});
            }
        }

        for(int tj=0;tj<BlocksX;tj++)
        {
            for(int i=0;i<tile_x;i++)
            {
                {{ routine.type_str }} temp=alpha*read_channel_intel({{ channels["channel_in_vector_x"] }});


                //here we read one row/column of A and multiply it for the same value of x
                for(int jj=0;jj<computing_outer_loop_limit;jj++)
                {
                    //receive elemnts of a: decoupling this form the computation loop
                    //maybe useful in case the sender of A does not perform unrolled writes into the channel
                    {{ routine.type_str }} local_A[WIDTH];
                    #pragma unroll
                    for(int j=0;j<WIDTH;j++)
                            local_A[j]=read_channel_intel({{ channels["channel_in_matrix_A"] }});

                    //updates all y

                    #pragma unroll
                    for(int j=0;j<WIDTH;j++)
                        local_y[jj*WIDTH+j]+=local_A[j]*temp;

                 }
            }
        }

        //now we can send this block of y (avoid the unroll since the tile size can be large)

        for(int i=0;i<tile_y;i++)
            write_channel_intel({{ channels["channel_out_vector"] }},local_y[i]);
	}

}

