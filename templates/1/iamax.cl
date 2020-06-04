/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2020 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    IAMAX finds the index of the first element having maximum absolute value

    Data is received from the input stream {{ channels["channel_in_vector_x"] }}.
    Data elements must be streamed with a padding equal to {{ routine.width }}
    (padding data must be set to zero).

    Result is streamed in an output channel at the end of the computation
    in a channel {{ channels["channel_out_scalar"] }}

*/
#pragma OPENCL EXTENSION cl_intel_channels : enable

channel {{ routine.type_str }} {{ channels["channel_in_vector_x"] }} __attribute__((depth({{ routine.width }})));
channel int {{ channels["channel_out_scalar"] }} __attribute__((depth(1)));


__kernel void {{ routine.user_name }}(int N)
{
    __constant uint WIDTH = {{ routine.width }};
    const int outer_loop_limit=1+(int)((N-1)/WIDTH); //ceiling
    {{ routine.type_str }} x[WIDTH];
    int g_max_index=0;
    {{ routine.type_str }} g_max_value=0;

    if(N>0)
    {
        for(int i=0; i<outer_loop_limit; i++)
        {
            int max_index;
            {{ routine.type_str }} max_value=0;

            #pragma unroll
            for(int j=0; j < WIDTH ;j++)
            {
                 x[j]=read_channel_intel({{ channels["channel_in_vector_x"] }});
                 if(fabs(x[j])>max_value)
                 {
                     max_index=i*WIDTH+j;
                     max_value=fabs(x[j]);
                 }
            }

            if(max_value>g_max_value)
            {
                g_max_index=max_index;
                g_max_value=max_value;
            }

        }
    }
    write_channel_intel({{ channels["channel_out_scalar"] }},g_max_index);

}
