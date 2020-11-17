/*

    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2020 ETH-Zurich. All rights reserved.
    See LICENSE for license information.


    ROT - applies a plan rotation

    Data is received from two input streams
    {{ channels["channel_in_vector_x"] }} and {{ channels["channel_in_vector_y"] }} having the proper type.
    Data elements must be streamed with a padding equal to W
    (padding data must be set to zero).

    Result is streamed in an output channel at the end of the computation
    in two channels  {{ channels["channel_out_vector_x"] }} (for x) and  {{ channels["channel_out_vector_y"] }} (for y)

*/


#pragma OPENCL EXTENSION cl_intel_channels : enable

{% if routine.type_str == 'double' %}
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
{% endif %}

channel {{ routine.type_str }} {{ channels["channel_in_vector_x"] }} __attribute__((depth({{ routine.width }})));
channel {{ routine.type_str }} {{ channels["channel_in_vector_y"] }} __attribute__((depth({{ routine.width }})));
channel {{ routine.type_str }} {{ channels["channel_out_vector_x"] }} __attribute__((depth({{ routine.width }})));
channel {{ routine.type_str }} {{ channels["channel_out_vector_y"] }} __attribute__((depth({{ routine.width }})));

__kernel void {{ routine.user_name }}(const int N, const {{ routine.type_str }} c, const {{ routine.type_str }} s)
{
    __constant uint WIDTH = {{ routine.width }};

    if(N==0) return;
    const int outer_loop_limit=1+(int)((N-1)/WIDTH); //ceiling
    {{ routine.type_str }} x[WIDTH];
    {{ routine.type_str }} y[WIDTH];
    {{ routine.type_str }} out_x[WIDTH];
    {{ routine.type_str }} out_y[WIDTH];

    for(int i=0; i<outer_loop_limit; i++)
    {
        #pragma unroll
        for(int j=0;j<WIDTH;j++)
        {
            x[j]=read_channel_intel({{ channels["channel_in_vector_x"] }});
            y[j]=read_channel_intel({{ channels["channel_in_vector_y"] }});

            out_x[j] = c * x[j] + s * y[j];
            out_y[j] = -s * x[j] + c * y[j];
            write_channel_intel({{ channels["channel_out_vector_x"] }},out_x[j]);
            write_channel_intel({{ channels["channel_out_vector_y"] }},out_y[j]);
        }
    }
}
