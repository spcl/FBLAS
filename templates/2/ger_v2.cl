/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2020 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    GER_V2 performs the rank 1 operation

    	A := alpha*x*y**T + A

    A arrives in tiles by Columns, Column Streamed.

    Data is received from three different channels ({{ channels["channel_in_vector_x"] }}, {{ channels["channel_in_vector_y"] }}
    and { channels["channel_in_matrix_A"] }}). Input data must be padded with zeros according to
    the reference tile sizes (TILE_N and TILE_M).

    Result is streamed in an output channel, tile by tile as soon as it is available.

    Check the kernel documentation for further information

*/

#pragma OPENCL EXTENSION cl_intel_channels : enable

{% if routine.type_str == 'double' %}
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
{% endif %}


channel {{ routine.type_str }} {{ channels["channel_in_vector_x"] }} __attribute__((depth({{ routine.width }})));
channel {{ routine.type_str }} {{ channels["channel_in_vector_y"] }} __attribute__((depth({{ routine.width }})));
channel {{ routine.type_str }} {{ channels["channel_in_matrix_A"] }} __attribute__((depth({{ routine.width }})));
channel {{ routine.type_str }} {{ channels["channel_out_matrix"] }} __attribute__((depth({{ routine.width }})));

/**
    This case can be used if A is column streamed
    - A is received by columns and tiles by column:
    - x must be received multiple times (M/TILE_M)
    - reuse is done over x and y
*/

__kernel void {{ routine.user_name }}(const {{ routine.type_str }} alpha, const int N, const int M )
{

    __constant uint WIDTH = {{ routine.width }};
    __constant uint TILE_N = {{ routine.tile_n_size }};
    __constant uint TILE_M = {{ routine.tile_m_size }};

    const int reading_x_outer_loop_limit=(int)(TILE_N/WIDTH);
    const int reading_y_outer_loop_limit=(int)(TILE_M/WIDTH);

    const int BlocksN=1+(int)((N-1)/TILE_N); //ceiling for padded data
    const int BlocksM=1+(int)((M-1)/TILE_M);
    const int computing_outer_loop_limit=(int)(TILE_N/WIDTH);

    {{ routine.type_str }} local_A[WIDTH];
    {{ routine.type_str }} local_x[TILE_N];
    {{ routine.type_str }} local_y[TILE_M];
    //Tiles are received by columns

    for(int tj=0;tj<BlocksM;tj++)
    {
        //in this case we reuse the corresponding block of y
        #pragma loop_coalesce
        for(int i=0;i<reading_y_outer_loop_limit;i++)
        {
            #pragma unroll
            for(int ii=0;ii<WIDTH;ii++)
            {
                local_y[i*WIDTH+ii]=read_channel_intel({{ channels["channel_in_vector_y"] }});
            }
        }
        #pragma loop_coalesce
        for(int ti=0; ti< BlocksN;ti++)
        {
            for(int j=0;j<TILE_M;j++)
            {
                for(int i=0;i<computing_outer_loop_limit;i++)
                {
                    //receive a column of A and compute
                    {{ routine.type_str }} tmp=alpha*local_y[j];
                    if(j==0)	//read the portion of x
                    {
                        #pragma unroll
                        for(int ii=0;ii<WIDTH;ii++)
                            local_x[i*WIDTH+ii]=read_channel_intel({{ channels["channel_in_vector_x"] }});
                    }
                    //read A and compute
                    #pragma unroll
                    for(int ii=0;ii<WIDTH;ii++)
                    {
                        local_A[ii]=tmp*local_x[i*WIDTH+ii]+read_channel_intel({{ channels["channel_in_matrix_A"] }});
                        write_channel_intel({{ channels["channel_out_matrix"] }},local_A[ii]);
                    }
                }
            }
        }

    }
}
