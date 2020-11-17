/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2020 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    Reads a vector of type {{ helper.type_str }} from memory and  push it
    into {{ channels["channel_out_vector"] }}. The vector is accessed with stride INCY.
    At first iterations it generates block 0, then block 0 and 1, ....
    Block size is given by macro TILE_N

    W memory reads are performed simultaneously. In the same way W channel pushes are performed
    at each clock cycle.  Data is padded to TILE_N using zero elements.

    It is used for routines SYR2
*/

{% if generate_channel_declaration is defined %}
#pragma OPENCL EXTENSION cl_intel_channels : enable

channel {{ helper.type_str }} {{ channels["channel_out_vector"] }} __attribute__((depth({{ helper.width }})));
{% endif %}

__kernel void {{ helper_name }}(__global volatile {{ helper.type_str }} *restrict data, unsigned int N)
{
    __constant uint WIDTH = {{ helper.width }};
    __constant int INCY = {{ helper.incy }};
    __constant uint TILE_N = {{ helper.tile_n_size }};
    const int BlocksN=1+(int)((N-1)/TILE_N);
    int outer_loop_limit=(int)(TILE_N/WIDTH);

    for(int ti=0; ti<BlocksN; ti++)
    {
        int offset=((INCY) > 0 ?  0 : ((N) - 1) * (-(INCY)));
        //send all the previous blocks
        //plus this one that1 we will receive back
        for(int tj=0;tj<=ti;tj++)
        {
            for(int i=0;i<outer_loop_limit;i++)
            {

                {{ helper.type_str }} y[WIDTH];
                //prepare data
                #pragma unroll
                for(int k=0;k<WIDTH;k++)
                {
                    if(i*WIDTH+k<N)
                        y[k]=data[offset+(k*INCY)];
                    else
                        y[k]=0;
                }
                offset+=WIDTH*INCY;
                //send data
                #pragma unroll
                for(int k=0;k<WIDTH;k++)
                    write_channel_intel({{ channels["channel_out_vector"] }},y[k]);
            }
        }
    }
}
