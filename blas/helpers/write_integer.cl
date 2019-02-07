/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2019 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    Reads an integer from CHANNEL_OUT and writes it into memory
    The name of the kernel can be redefined by means of preprocessor MACROS.
*/

__kernel void WRITE_SCALAR(__global int *restrict out)
{
        *out = read_channel_intel(CHANNEL_OUT);
}
