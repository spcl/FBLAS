/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2019 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    SCAL scales a vector by a constant.

    Data is received through an input channel {{ channels["channel_in_vector_x"] }}.
    Results are produced into the output channel {{ channels["channel_out_vector"] }}.
    Data must arrive (and it is produced) padded with size {{ routine.width }}.
    Padding data must be set (or is set) to zero

*/

#pragma OPENCL EXTENSION cl_intel_channels : enable

{% if routine.type_str == 'double' %}
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
{% endif %}

channel {{ routine.type_str }} {{ channels["channel_in_vector_x"] }} __attribute__((depth({{ routine.width }})));
channel {{ routine.type_str }} {{ channels["channel_out_vector"] }} __attribute__((depth({{ routine.width }})));



__kernel void {{ routine.user_name }}(unsigned int N, {{ routine.type_str }} alpha)
{
    __constant uint WIDTH = {{ routine.width }};
    const int outer_loop_limit=1+(int)((N-1)/WIDTH); //ceiling
    {{ routine.type_str }} x[WIDTH];

    for(int i=0; i<outer_loop_limit; i++)
    {
        #pragma unroll
        for(int j=0;j<WIDTH;j++)
        {
            x[j]=alpha*read_channel_intel({{ channels["channel_in_vector_x"] }});
            write_channel_intel({{ channels["channel_out_vector"] }},x[j]);
        }

    }

}
