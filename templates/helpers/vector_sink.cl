/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2020 ETH-Zurich. All rights reserved.
    See LICENSE for license information.


    Reads a vector of type {{ helper.type_str }} from channel {{ channels["channel_in_vector"] }}.
    The name of the kernel can be redefined by means of preprocessor MACROS.
    INCW represent the access stride.

    WIDTH reads are performed simultaneously.
    Data arrives padded at pad_size.
*/

{% if generate_channel_declaration is defined %}
#pragma OPENCL EXTENSION cl_intel_channels : enable

channel {{ helper.type_str }} {{ channels["channel_in_vector"] }} __attribute__((depth({{ helper.width }})));
{% endif %}

__kernel void {{ helper_name }}(__global volatile {{ helper.type_str }}  *restrict out, unsigned int N, unsigned int pad_size)
{
    __constant uint WIDTH = {{ helper.width }};
    const unsigned int ratio=pad_size/WIDTH;
    const unsigned int padding_loop_limit=ceil(((float)N)/pad_size);
    const unsigned int outer_loop_limit=padding_loop_limit*ratio;
    //receive data
    for(int i=0;i<outer_loop_limit;i++)
    {
        #pragma unroll
        for(int j=0;j<WIDTH;j++)
        {
            read_channel_intel({{ channels["channel_in_vector"] }});
        }
    }
}
