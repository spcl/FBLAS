/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2020 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    Generates a dummy vector  of type TYPE_T without reading from memory and  push it
    into {{ channels["channel_out_vector"] }}.

    W={{ helper.width}} memory reads are performed simultaneously. In the same way W channel push are performed.
    Data is padded to pad_size. Pad_size must be a multiple of WIDTH.
    So, for level 1 routines pad_size will be equal to W.
    For level 2, in which you have to respect some tiling, it will be equal to a tile size.

    Data is padded using zero elements.

    The vector is sent 'repetitions' times
*/

{% if generate_channel_declaration is defined%}
#pragma OPENCL EXTENSION cl_intel_channels : enable

channel {{ helper.type_str }} {{ channels["channel_out_vector"] }} __attribute__((depth({{ helper.width }})));
{% endif %}

__kernel void {{ helper_name }}(unsigned int N, unsigned int repetitions)
{
    __constant uint WIDTH = {{ helper.width }};

    const unsigned int outer_loop_limit=1+((int)((N-1)/WIDTH));

    #pragma loop_coalesce
    for(int t=0; t< repetitions;t++)
    {
        for(int i=0;i<outer_loop_limit;i++)
        {
            {{ helper.type_str }} x[WIDTH];
            //prepare data
            #pragma unroll
            for(int k=0;k<WIDTH;k++)
            {
                if(i*WIDTH+k<N)
                    x[k]=i*WIDTH+k;
                else
                    x[k]=0;
            }

            //send data
            #pragma unroll
            for(int k=0;k<WIDTH;k++)
                write_channel_intel({{ channels["channel_out_vector"] }},x[k]);
        }
    }
}
