/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2020 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    Reads a scalar vector of type {{ helper.type_str }}  and writes it into memory

*/

{% if generate_channel_declaration is defined%}
#pragma OPENCL EXTENSION cl_intel_channels : enable

channel {{ helper.type_str }} {{ channels["channel_in_scalar"] }} __attribute__((depth(1)));
{% endif %}

__kernel void {{ helper_name}}(__global volatile {{ helper.type_str }} *restrict out)
{
        *out = read_channel_intel({{ channels["channel_in_scalar"] }});
}
