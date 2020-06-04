/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2019 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    Reads a vector of type {{ helper.type_str }} from memory and  push it
    into {{ channels["channel_out_vector"] }}. The vector is accessed with {{ helper.incy }} INCY.
    The name of the kernel can be redefined by means of preprocessor MACROS.

    W={{ helper.width}} memory reads are performed simultaneously. In the same way W channel push are performed.
    Data is padded to pad_size. Pad_size must be a multiple of W.
    So, for level 1 routines pad_size will be probably equal to W.
    For level 2, in which you have to respect some tiling, it will be equal to a tile size.

    Data is padded to W using zero elements.

    The vector is sent 'repetitions' times.
*/

{% if generate_channel_declaration is defined%}
#pragma OPENCL EXTENSION cl_intel_channels : enable

channel {{ helper.type_str }} {{ channels["channel_out_vector"] }} __attribute__((depth({{ helper.width }})));
{% endif %}

__kernel void {{ helper_name}}(__global volatile {{ helper.type_str }} *restrict data, unsigned int N, unsigned int pad_size, unsigned int repetitions)
{
    __constant uint WIDTH = {{ helper.width }};
    __constant int INCY = {{ helper.incy }};

    const unsigned int ratio=pad_size/WIDTH;
    const unsigned int padding_loop_limit=ceil(((float)N)/pad_size);
    const unsigned int outer_loop_limit=padding_loop_limit*ratio;

    #pragma loop_coalesce
    for(int t=0; t< repetitions;t++)
    {
        //compute the starting index
        int offset=((INCY) > 0 ?  0 : ((N) - 1) * (-(INCY)));
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