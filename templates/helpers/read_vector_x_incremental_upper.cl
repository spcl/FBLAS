/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2020 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    Reads a vector of type {{ helper.type_str }} from memory and  push it
    into {{ channels["channel_out_vector"] }}. The vector is accessed with stride INCX.
    At first iterations it generates block 0 to blocksN,  then block1 to N, ....
    Block size is given by TILE_N.

    The name of the kernel can be redefined by means of preprocessor MACROS.

    W memory reads are performed simultaneously. In the same way W channel push are performed.
    Data is padded to TILE_N.
    So, for level 1 routines pad_size will be  equal to W.
    Data is padded using zero elements.

    It is used for routines SYR and SYR2
*/


{% if generate_channel_declaration is defined %}
#pragma OPENCL EXTENSION cl_intel_channels : enable

channel {{ helper.type_str }} {{ channels["channel_out_vector"] }} __attribute__((depth({{ helper.width }})));
{% endif %}


__kernel void {{ helper_name }}(__global {{ helper.type_str }} *restrict data, unsigned int N)
{
    __constant uint WIDTH = {{ helper.width }};
    __constant int INCX = {{ helper.incx }};
    __constant uint TILE_N = {{ helper.tile_n_size }};
    const int BlocksN=1+(int)((N-1)/TILE_N);
    int outer_loop_limit=(int)(TILE_N/WIDTH);

    for(int ti=0; ti<BlocksN; ti++)
    {
        int offset=((INCX) > 0 ?  0 : ((N) - 1) * (-(INCX)));

        //send the curretn block and all the following
        for(int tj=ti;tj<BlocksN;tj++)
        {
            for(int i=0;i<outer_loop_limit;i++)
            {
                {{ helper.type_str }} x[WIDTH];
                //prepare data
                #pragma unroll
                for(int k=0;k<WIDTH;k++)
                {
                    if(i*WIDTH+k<N)
                        x[k]=data[ti*TILE_N*INCX+offset+(k*INCX)]; //we have to take the upper part of x
                    else
                        x[k]=0;
                }
                offset+=WIDTH*INCX;
                //send data
                #pragma unroll
                for(int k=0;k<WIDTH;k++)
                    write_channel_intel({{ channels["channel_out_vector"] }},x[k]);

            }
        }
    }

}
