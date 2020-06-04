/**

    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2020 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    NRM2 returns the euclidean norm of a vector via the function
    name, so that
            NRM2 := sqrt( x'*x )

    Data is received from the input stream {{ channels["channel_in_vector_x"] }}.
    Data elements must be streamed with a padding equal to {{ routine.width }}
    (padding data must be set to zero).

    Result is streamed in an output channel at the end of the computation
    in a channel {{ channels["channel_out_scalar"] }}.

    If DOUBLE_PRECISION is defined, the routine computes DNRM2
    otherwise SRNM2

*/

#pragma OPENCL EXTENSION cl_intel_channels : enable

{% if routine.type_str == 'double' %}
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
{% endif %}

channel {{ routine.type_str }} {{ channels["channel_in_vector_x"] }} __attribute__((depth({{ routine.width }})));
channel {{ routine.type_str }} {{ channels["channel_out_scalar"] }} __attribute__((depth(1)));


__kernel void {{ routine.user_name }}(int N)
{
    __constant uint WIDTH = {{ routine.width }};
    {% if routine.uses_shift_registers %}
    __constant uint SHIFT_REG = {{ routine.size_shift_registers }};
    {% endif %}

    const int outer_loop_limit=1+(int)((N-1)/WIDTH); //ceiling
    {{ routine.type_str }} acc_o=0;
    {{ routine.type_str }} x[WIDTH];

    {% if routine.uses_shift_registers %}
    {{ routine.type_str }} shift_reg[SHIFT_REG+1]; //shift register

    #pragma unroll
    for(int i=0;i<SHIFT_REG+1;i++)
       shift_reg[i]=0;
    {% endif %}

    for(int i=0; i<outer_loop_limit; i++)
    {
        //By unrolling this two loops, we perform W mult. per cycle
        {{ routine.type_str }} acc_i=0;
        #pragma unroll
        for(int j=0;j<WIDTH;j++)
        {
            x[j]=read_channel_intel({{ channels["channel_in_vector_x"] }});
            acc_i+=x[j]*x[j];
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
    #pragma unroll
    for(int i=0;i<SHIFT_REG;i++)
        acc_o+=shift_reg[i];
    {% endif %}

    {{ routine.type_str }} res=sqrt(acc_o);
    write_channel_intel({{ channels["channel_out_scalar"] }},res);

}
