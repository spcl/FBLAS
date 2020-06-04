/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2020 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    GER_V3 performs the rank 1 operation

    A := alpha*x*y**T + A

    A arrives in tiles by cols, Row Streamed.

    Data is received from three different channels ({{ channels["channel_in_vector_x"] }}, {{ channels["channel_in_vector_y"] }}
    and {{ channels["channel_in_matrix_A"] }}). Input data must be padded with zeros according to
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
    This case can be used if A is row streamed, tiles are sent in column order
    - x must be received multiple times (M/TILE_M)
    - reuse is done over x and y
    - A row streamed, tiles by columns
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
    const int computing_outer_loop_limit=(int)(TILE_M/WIDTH);


    {{ routine.type_str }} local_A[WIDTH];
    {{ routine.type_str }} local_x[TILE_N];
    {{ routine.type_str }} local_y[TILE_M];
    //Tiles are received by columns

    //#pragma loop_coalesce
    for(int tj=0;tj<BlocksM;tj++)
    {

        //in this case we reuse the block of y
        for(int i=0;i<reading_y_outer_loop_limit;i++)
        {
            #pragma unroll
            for(int ii=0;ii<WIDTH;ii++)
            {
                local_y[i*WIDTH+ii]=read_channel_intel({{ channels["channel_in_vector_y"] }});
            }
        }
        #pragma loop_coalesce 2
        for(int ti=0; ti< BlocksN;ti++)
        {
            //read X
            /*for(int i=0;i<reading_x_outer_loop_limit;i++)
            {
                #pragma unroll
                for(int ii=0;ii<WIDTH;ii++)
                    local_x[i*WIDTH+ii]=
            }*/

            for(int i=0;i<TILE_N;i++)
            {
                //receive a row of A and compute
                {{ routine.type_str }} tmp=alpha*read_channel_intel({{ channels["channel_in_vector_x"] }});
                for(int j=0;j<computing_outer_loop_limit;j++)
                {

                    /*if(i ==0 && ti ==0){
                        #pragma unroll
                        for(int ii=0;ii<WIDTH;ii++)
                        {
                            local_y[j*WIDTH+ii]=read_channel_intel({{ channels["channel_in_vector_y"] }});
                        }
                    }*/

                    #pragma unroll
                    for(int jj=0;jj<WIDTH;jj++)
                    {
                        local_A[jj]=tmp*local_y[j*WIDTH+jj]+read_channel_intel({{ channels["channel_in_matrix_A"] }});
                        write_channel_intel({{ channels["channel_out_matrix"] }},local_A[jj]);
                    }
                }
            }
        }
    }
}
