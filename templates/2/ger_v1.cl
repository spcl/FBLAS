/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2020 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    GER V1 performs:

    A := alpha*x*y**T + A

    A arrives in tiles by rows, Row Streamed.

    Data is received from three different channels ({{ channels["channel_in_vector_x"] }}, {{ channels["channel_in_vector_y"] }}
    and {{ channels["channel_in_matrix_A"] }} A). Input data must be padded with zeros according to
    the reference tile sizes (TILE_N and TILE_M).

    Result is streamed in an output channel, tile by tile as soon as it is available,
    respecting the same order of arrival of the input matrix

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


/*
    This version:
    - A by rows, tiles of size TILE_N x TILE_M by rows, elements in row order. y must be sent entirely N/Tile_N times
        Reuse is done over x by keeping a blocks of size TILE_N and over y by keeping a blocks of size TILE_M
*/

__kernel void {{ routine.user_name }}(const {{ routine.type_str }} alpha, const int N, const int M )
{

    __constant uint WIDTH = {{ routine.width }};
    __constant uint TILE_N = {{ routine.tile_n_size }};
    __constant uint TILE_M = {{ routine.tile_m_size }};

    //loops for buffering x and y are unrolled
    const int reading_x_outer_loop_limit=(int)(TILE_N/WIDTH);
    const int reading_y_outer_loop_limit=(int)(TILE_M/WIDTH);

    const int BlocksN=1+(int)((N-1)/TILE_N); //ceiling for padded data
    const int BlocksM=1+(int)((M-1)/TILE_M);
    const int computing_outer_loop_limit=(int)(TILE_M/WIDTH);

    {{ routine.type_str }} local_A[WIDTH];
    {{ routine.type_str }} local_x[TILE_N];
    {{ routine.type_str }} local_y[TILE_M];

    for(int ti=0; ti<BlocksN;ti++)
    {
        //here we reuse a block of x
        #pragma loop_coalesce
        for(int i=0;i<reading_x_outer_loop_limit;i++)
        {
            #pragma unroll
            for(int ii=0;ii<WIDTH;ii++)
                local_x[i*WIDTH+ii]=read_channel_intel({{ channels["channel_in_vector_x"] }});
        }
        #pragma loop_coalesce 2
        for(int tj=0;tj<BlocksM;tj++)
        {
            for(int i=0;i<TILE_N;i++)
            {
                //can not coalesce more due to variable length loop
                if(i==0){//read y
                    for(int k=0;k<reading_y_outer_loop_limit;k++)
                    {
                        #pragma unroll
                        for(int kk=0;kk<WIDTH;kk++)
                            local_y[k*WIDTH+kk]=read_channel_intel({{ channels["channel_in_vector_y"] }});
                    }
                }
                for(int j=0;j<computing_outer_loop_limit;j++)
                {
                    {{ routine.type_str }} tmp=alpha*local_x[i];

                    #pragma unroll
                    for(int jj=0;jj<WIDTH;jj++)
                    {
                        //compute and send out the data
                        local_A[jj]=tmp*local_y[j*WIDTH+jj]+read_channel_intel({{ channels["channel_in_matrix_A"] }});
                        write_channel_intel({{ channels["channel_out_matrix"] }},local_A[jj]);
                    }
                }
            }
        }
    }
}
