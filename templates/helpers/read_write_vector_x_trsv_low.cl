/*
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2020 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    This helper is intended to be used with trsv

    Reads a vector of type {{ helper.type_str }} from memory and  push it
    into {{ channels["channel_out_vector"] }}.
    The vector is sent in blocks and each block may be sent multiple times.
    At the first iteration it sends the first block. Then the first and the second, ...
    At iteration i, it sends blocks from O to i-1.
    At each iteration, after sending blocks i, it gets back the same block updated
    from {{ channels["channel_in_vector"] }} 

    The vector is accessed with stride INCX.

    WIDTH memory reads are performed simultaneously. In the same way WIDTH channel pushes are performed .
    Data is padded to TILE_N using zero elements.

*/

{% if generate_channel_declaration is defined %}
#pragma OPENCL EXTENSION cl_intel_channels : enable

channel {{ helper.type_str }} {{ channels["channel_out_vector"] }} __attribute__((depth({{ helper.width }})));
channel {{ helper.type_str }} {{ channels["channel_in_vector"] }} __attribute__((depth({{ helper.width }})));

{% endif %}

__kernel void {{ helper_name }}(__global volatile {{ helper.type_str }} *restrict data, int N)
{
    __constant uint WIDTH = {{ helper.width }};
    __constant uint TILE_N = {{ helper.tile_n_size }};
    __constant int INCX = {{ helper.incx }};

    const int BlocksN=1+(int)((N-1)/TILE_N);
    int outer_loop_limit=(int)(TILE_N/WIDTH);


    for(int ti=0; ti<BlocksN; ti++)
    {
        //send all the previous blocks
        //plus this one that we will receive back
        int offset=((INCX) > 0 ?  0 : ((N) - 1) * (-(INCX)));

        for(int tj=0;tj<=ti;tj++)
        {
            for(int i=0;i<outer_loop_limit;i++)
            {
                {{ helper.type_str }} x[WIDTH];
                //prepare data
                #pragma unroll
                for(int k=0;k<WIDTH;k++)
                {
                    if(i*WIDTH+k<N)
                        x[k]=data[offset+(k*INCX)];
                    else
                        x[k]=0;
                }
                offset+=WIDTH*INCX;

                #pragma unroll
                for(int ii=0; ii<WIDTH ;ii++)
                {
                    write_channel_intel({{ channels["channel_out_vector"] }},x[ii]);
                }
            }
        }

       //get back the result and overwrite it
       offset -= TILE_N*INCX;
       for(int i=0;i<outer_loop_limit;i++)
       {
           #pragma unroll
           for(int ii=0; ii<WIDTH ;ii++)
           {
                {{ helper.type_str }} r=read_channel_intel({{ channels["channel_in_vector"] }} );
                if(ti*TILE_N+i*WIDTH+ii<N)
                    data[offset+(ii*INCX)]=r;
           }
           offset+=WIDTH*INCX;
       }
    }

}
