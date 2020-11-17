 /**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2020 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    DOT performs the dot product of two vectors.

    Streamed version: data is received from two input streams
    {{ channels["channel_in_vector_x"] }} and {{ channels["channel_in_vector_y"] }} having the proper type.
    Data elements must be streamed with a padding equal to W
    (padding data must be set to zero).

    Result is streamed in an output channel at the end of the computation
    in a channel {{ channels["channel_out_scalar"] }}

*/

#pragma OPENCL EXTENSION cl_intel_channels : enable

{% if routine.type_str == 'double' %}
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
{% endif %}


channel {{ routine.type_str }} {{ channels["channel_in_vector_x"] }} __attribute__((depth({{ routine.width }})));
channel {{ routine.type_str }} {{ channels["channel_in_vector_y"] }} __attribute__((depth({{ routine.width }})));
channel {{ routine.type_str }} {{ channels["channel_out_scalar"] }} __attribute__((depth(1)));


/**
    Performs streaming dot product: data is received through
    {{ channels["channel_in_vector_x"] }} and {{ channels["channel_in_vector_y"] }}. Result is sent
    to {{ channels["channel_out_scalar"] }}.
*/
__kernel void {{ routine.user_name }}(int N)
{
    __constant uint WIDTH = {{ routine.width }};
    {% if routine.uses_shift_registers %}
    __constant uint SHIFT_REG = {{ routine.size_shift_registers }};
    {% endif %}


    {{ routine.type_str }} acc_o=0;
    if(N>0)
    {

        const int outer_loop_limit=1+(int)((N-1)/WIDTH); //ceiling
        {{ routine.type_str }} x[WIDTH],y[WIDTH];

        {% if routine.uses_shift_registers %}
        {{ routine.type_str }} shift_reg[SHIFT_REG+1]; //shift register

        #pragma unroll
        for(int i=0;i<SHIFT_REG+1;i++)
           shift_reg[i]=0;
        {% endif %}

        //Strip mine the computation loop to exploit unrolling
        for(int i=0; i<outer_loop_limit; i++)
        {

            {{ routine.type_str }} acc_i=0;
            #pragma unroll
            for(int j=0;j<WIDTH;j++)
            {
                x[j]=read_channel_intel({{ channels["channel_in_vector_x"] }});
                y[j]=read_channel_intel({{ channels["channel_in_vector_y"] }});
                acc_i+=x[j]*y[j];

            }

            {% if routine.uses_shift_registers %}

                shift_reg[SHIFT_REG] = shift_reg[0]+acc_i;
                //Shift every element of shift register
                #pragma unroll
                for(int j = 0; j < SHIFT_REG; ++j)
                    shift_reg[j] = shift_reg[j + 1];
            {% else %}
                acc_o+=acc_i;
            {% endif %}

        }

        {% if routine.uses_shift_registers %}
            //reconstruct the result using the partial results in shift register
            {{ routine.type_str }} acc=0;
            #pragma unroll
            for(int i=0;i<SHIFT_REG;i++)
                acc+=shift_reg[i];
            acc_o = acc;
        {% endif %}
    }
    else //no computation: result is zero
        acc_o=0.0f;
    //write to the sink
    write_channel_intel({{ channels["channel_out_scalar"] }},acc_o);
}
