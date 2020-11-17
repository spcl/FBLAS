/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2020 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    AXPY constant times a vector plus a vector.

    Streamed version: data is received from two input streams
    {{ channels["channel_in_vector_x"] }} and {{ channels["channel_in_vector_y"] }} having the proper type.
    Data elements must be streamed with a padding equal to W
    (padding data must be set to zero).

    Result is streamed in an output channel at the end of the computation
    in a channel {{ channels["channel_out_vector"] }}

*/


#pragma OPENCL EXTENSION cl_intel_channels : enable

{% if routine.type_str == 'double' %}
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
{% endif %}


channel {{ routine.type_str }} {{ channels["channel_in_vector_x"] }} __attribute__((depth({{ routine.width }})));
channel {{ routine.type_str }} {{ channels["channel_in_vector_y"] }} __attribute__((depth({{ routine.width }})));
channel {{ routine.type_str }} {{ channels["channel_out_vector"] }} __attribute__((depth({{ routine.width }})));

__kernel void {{ routine.user_name }}(const {{ routine.type_str }}  alpha, int N)
{
    __constant uint WIDTH = {{ routine.width }};
    if(N==0) return;

    const int outer_loop_limit=1+(int)((N-1)/WIDTH); //ceiling
    {{ routine.type_str }}  res[WIDTH];

    for(int i=0; i<outer_loop_limit; i++)
    {
        //receive WIDTH elements from the input channels
        #pragma unroll
        for(int j=0;j<WIDTH;j++)
            res[j]=alpha*read_channel_intel({{ channels["channel_in_vector_x"] }})+read_channel_intel({{ channels["channel_in_vector_y"] }});

        //sends the data to a writer
        #pragma unroll
        for(int j=0; j<WIDTH; j++)
            write_channel_intel({{ channels["channel_out_vector"] }},res[j]);
    }

}
